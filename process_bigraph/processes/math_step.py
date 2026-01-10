import sys
import inspect
import pytest

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from process_bigraph import allocate_core
from process_bigraph.composite import Step, Process, Composite
from process_bigraph.emitter import emitter_from_wires


class Tick(Process):
    config_schema = {"dt": {"_type":"float", "_default":1.0}}

    def inputs(self):
        return {"t": "float"}

    def outputs(self):
        return {"t": "overwrite[float]"}

    def update(self, state, interval):
        return {"t": float(state["t"]) + float(interval)}


# ---------- the Step ----------
class MathExpressionStep(Step):
    """
    Evaluate a set of user-configured mathematical expressions as a single Step.

    This Step parses a list of named expressions, infers the required input ports
    from the expressions' free symbols, and creates output ports for each named
    expression. Expressions may reference other outputs; the Step will compute a
    dependency-respecting evaluation order automatically via a topological sort.

    Key behaviors
    -------------
    - Dynamic ports:
        * Outputs are exactly the set of `out` names in `config["expressions"]`.
        * Inputs are any free symbols that are not output names and not parameters.
    - Dependency ordering:
        * If an expression references another output, it becomes a dependency.
        * Expressions are evaluated in topological order, not list order.
        * Cycles (algebraic loops) raise a ValueError with a dependency hint.
    - Compilation:
        * Each expression is parsed with SymPy and compiled once in `initialize()`
          using `sympy.lambdify` with the selected backend (default: NumPy).
        * During `update()`, expressions are evaluated using current state inputs
          and any outputs computed earlier in the dependency order.
    - Semantics:
        * Returns overwrite patches for all outputs computed during `update()`.

    Configuration
    -------------
    expressions : list[dict]
        List of {"out": <str>, "expr": <str>} entries. Each `out` becomes an
        output port and state field.
    params : dict[str, float], optional
        Named constants available to expressions without becoming input ports.
    functions : str, optional
        SymPy lambdify backend module name (e.g. "numpy").
    debug : bool, optional
        If True, prints inferred ports, dependency order, and compilation details.

    Notes
    -----
    - This Step performs a single-pass evaluation in dependency order; it is not
      a simultaneous equation solver. Cyclic dependencies require a different
      approach (e.g., fixed-point iteration or root finding).
    """

    config_schema = {
        "expressions": "node",  # list[{"out": str, "expr": str}]
        "params": {"_type": "node", "_default": {}},
        "functions": {"_type": "string", "_default": "numpy"},
        "debug": {"_type": "boolean", "_default": True},
    }

    def initialize(self, config=None):
        cfg = self.config
        debug = bool(cfg.get("debug", True))

        expr_specs = cfg.get("expressions", None)
        if not isinstance(expr_specs, (list, tuple)) or len(expr_specs) == 0:
            raise ValueError("MathExpressionStep requires config['expressions'] as a non-empty list")

        self._params = dict(cfg.get("params", {}))
        backend = cfg.get("functions", "numpy")

        # Validate + parse
        out_names = []
        expr_map = {}  # out -> sympy expr
        raw_map = {}   # out -> expr string (for debugging)
        for spec in expr_specs:
            out = spec.get("out", None)
            expr_str = spec.get("expr", None)
            if not isinstance(out, str) or not out:
                raise ValueError("Each expression spec must have a non-empty string 'out'")
            if not isinstance(expr_str, str) or not expr_str:
                raise ValueError(f"Expression for '{out}' must be a non-empty string")
            if out in expr_map:
                raise ValueError(f"Duplicate output name: '{out}'")

            try:
                expr = sp.sympify(expr_str)
            except Exception as e:
                raise ValueError(f"Failed to parse expression for '{out}': {expr_str}\n{e}") from e

            out_names.append(out)
            expr_map[out] = expr
            raw_map[out] = expr_str

        self._out_names = out_names
        out_set = set(out_names)
        param_set = set(self._params.keys())

        # Build dependency graph among outputs
        # deps[out] = set of output-names that out depends on
        deps = {out: set() for out in out_names}
        external_inputs = set()

        for out, expr in expr_map.items():
            free = {str(s) for s in expr.free_symbols}

            # outputs referenced become dependencies
            deps[out] = {name for name in free if name in out_set and name != out}

            # anything else that isn't a param and isn't an output is an external input port
            external_inputs |= {name for name in free if name not in out_set and name not in param_set}

        # Topological sort (Kahn) to get execution order
        indeg = {o: 0 for o in out_names}
        for o in out_names:
            for d in deps[o]:
                indeg[o] += 1

        # Start with nodes that have no dependencies.
        # Use sorted() for deterministic order.
        queue = [o for o in sorted(out_names) if indeg[o] == 0]
        exec_order = []

        # adjacency: who depends on me
        rev = {o: set() for o in out_names}
        for o in out_names:
            for d in deps[o]:
                rev[d].add(o)

        while queue:
            n = queue.pop(0)
            exec_order.append(n)
            for m in sorted(rev[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)

        if len(exec_order) != len(out_names):
            # cycle exists
            remaining = [o for o in out_names if o not in exec_order]
            # give a helpful hint about the cycle region
            cycle_hint = {o: sorted(list(deps[o])) for o in remaining}
            raise ValueError(
                "Cyclic dependency between expressions. Cannot order outputs.\n"
                f"Remaining outputs: {remaining}\n"
                f"Dependencies among remaining: {cycle_hint}"
            )

        self._exec_order = exec_order
        self._in_names = sorted(external_inputs)

        # Compile each expression with *just the symbols it needs* (deterministic arg list)
        self._compiled = {}         # out -> callable
        self._needed_symbols = {}   # out -> list[str] (argument names)
        for out in out_names:
            expr = expr_map[out]
            needed = sorted({str(s) for s in expr.free_symbols})
            fn = sp.lambdify([sp.Symbol(n) for n in needed], expr, modules=backend)
            self._compiled[out] = fn
            self._needed_symbols[out] = needed

        if debug:
            print("\n[MathExpressionStep] initialize()")
            print("  outputs declared:", self._out_names)
            print("  params:", self._params)
            print("  inferred input ports:", self._in_names)
            print("  dependency order:", self._exec_order)
            for out in self._exec_order:
                print(f"   - {out} = {raw_map[out]}")
                print(f"     depends on outputs: {sorted(list(deps[out]))}")
                print(f"     needs symbols     : {self._needed_symbols[out]}")
            print("[MathExpressionStep] initialize complete\n")

        return cfg

    def inputs(self):
        return {name: "float" for name in self._in_names}

    def outputs(self):
        # all declared outputs are ports, regardless of execution order
        return {name: "overwrite[float]" for name in self._out_names}

    def update(self, state):
        # Values available for expression evaluation
        values = {}

        # External input ports come from state
        for name in self._in_names:
            values[name] = float(state[name])

        # Params are constants
        for k, v in self._params.items():
            values[k] = float(v)

        # Evaluate in dependency order
        out_patch = {}
        for out in self._exec_order:
            needed = self._needed_symbols[out]
            args = []
            for name in needed:
                if name not in values:
                    raise ValueError(
                        f"While computing '{out}', missing symbol '{name}'. "
                        f"(Likely an internal dependency not yet computed or missing input/param.)"
                    )
                args.append(values[name])

            y = self._compiled[out](*args)
            y = float(y)
            values[out] = y
            out_patch[out] = y

        return out_patch



def make_test_core():
    members = dict(inspect.getmembers(sys.modules[__name__]))
    return allocate_core(
        top=members)


@pytest.fixture
def core():
    return make_test_core()

def plot_single_eval(state, *, inputs=("a", "b", "c"), outputs=("z",), expected=None, title=None):
    """
    Plot a single evaluation of MathExpressionStep: inputs and outputs as bars.

    Parameters
    ----------
    state : dict-like
        Typically sim.state (or a nested dict) containing the variables.
    inputs, outputs : tuple[str]
        Names to plot from the state.
    expected : dict[str, float] | None
        Optional expected values to overlay for outputs, e.g. {"z": ...}.
    title : str | None
        Plot title.
    """
    names = list(inputs) + list(outputs)
    values = [float(state[n]) for n in names]

    plt.figure()
    plt.bar(names, values)
    plt.ylabel("value")
    plt.title(title or "Single evaluation: inputs → outputs")
    plt.show()

    if expected:
        # Small companion plot: output vs expected
        out_names = list(outputs)
        out_vals = [float(state[n]) for n in out_names]
        exp_vals = [float(expected[n]) for n in out_names]

        x = np.arange(len(out_names))
        width = 0.35

        plt.figure()
        plt.bar(x - width/2, out_vals, width, label="computed")
        plt.bar(x + width/2, exp_vals, width, label="expected")
        plt.xticks(x, out_names)
        plt.ylabel("value")
        plt.title("Outputs: computed vs expected")
        plt.legend()
        plt.show()

def plot_timeseries(records, series, *, x="t", title=None, xlabel=None, ylabel=None,
                    marker=".", linewidth=None, legend=True):
    if not records:
        raise ValueError("No records to plot")

    ts = np.array([r[x] for r in records], dtype=float)

    if isinstance(series, dict):
        keys = list(series.keys())
        labels = series
    else:
        keys = list(series)
        labels = {k: k for k in keys}

    plt.figure()
    for k in keys:
        ys = np.array([r[k] for r in records], dtype=float)
        plt.plot(ts, ys, marker=marker, linewidth=linewidth, label=labels[k])

    plt.xlabel(xlabel or x)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend and len(keys) > 1:
        plt.legend()
    plt.show()


def run_math_step_tick_only(core, total_time=12.0, dt=0.1):
    MATH_ADDR = f"local:!{MathExpressionStep.__module__}.MathExpressionStep"
    TICK_ADDR = f"local:!{Tick.__module__}.Tick"
    print("Using addresses:", {"tick": TICK_ADDR, "math": MATH_ADDR})

    sim = Composite(
        {
            "state": {
                "t": 0.0,
                "a": 0.0, "b": 0.0, "c": 0.0, "k": 0.0,
                "z": 0.0, "w": 0.0, "u": 0.0, "energy": 0.0,

                "tick": {
                    "_type": "process",
                    "address": TICK_ADDR,
                    "config": {},
                    "interval": dt,
                    "inputs": {"t": ["t"]},
                    "outputs": {"t": ["t"]},
                },

                "math": {
                    "_type": "step",
                    "address": MATH_ADDR,
                    "config": {
                        "expressions": [
                            {"out": "a", "expr": "2.0 + 1.2*sin(2*pi*0.35*t)"},
                            {"out": "b", "expr": "3.0 + 0.8*cos(2*pi*0.10*t + 0.4)"},
                            {"out": "c", "expr": "0.5 + 0.6*sin(2*pi*0.04*t)"},
                            {"out": "k", "expr": "1.0 + 2.0*(0.5 + 0.5*sin(2*pi*0.06*t))"},
                            {"out": "z", "expr": "a*b + sin(c)"},
                            {"out": "w", "expr": "z / (k + 1)"},
                            {"out": "u", "expr": "exp(-0.12*t) * cos(2*pi*0.55*t)"},
                            {"out": "energy", "expr": "0.5*(a**2 + b**2) + w**2 + 0.25*u**2"},
                        ],
                        "params": {"pi": float(np.pi)},
                        "functions": "numpy",
                        "debug": False,
                    },
                    "inputs": {"t": ["t"]},
                    "outputs": {
                        "a": ["a"], "b": ["b"], "c": ["c"], "k": ["k"],
                        "z": ["z"], "w": ["w"], "u": ["u"], "energy": ["energy"],
                    },
                },

                "emitter": emitter_from_wires(
                    {"t": ["t"], "a": ["a"], "b": ["b"], "c": ["c"], "k": ["k"],
                     "z": ["z"], "w": ["w"], "u": ["u"], "energy": ["energy"]}
                ),
            }
        },
        core=core,
    )

    sim.run(total_time)
    records = sim.state["emitter"]["instance"].query()

    print("n records:", len(records))
    print("first record:", records[0])
    print("last record :", records[-1])

    plot_timeseries(records, ["a", "b", "k"], title="Time-driven signals: a(t), b(t), k(t)")
    plot_timeseries(records, ["z"], title="z(t) = a(t)*b(t) + sin(c(t))", legend=False)
    plot_timeseries(records, ["w"], title="w(t) = z(t)/(k(t)+1)", legend=False)
    plot_timeseries(records, ["u"], title="u(t) damped oscillator", legend=False)
    plot_timeseries(records, ["energy"], title="energy(t)", legend=False)


def run_math_step_tick_only(core, total_time=12.0, dt=0.1):
    MATH_ADDR = f"local:!{MathExpressionStep.__module__}.MathExpressionStep"
    TICK_ADDR = f"local:!{Tick.__module__}.Tick"
    print("Using addresses:", {"tick": TICK_ADDR, "math": MATH_ADDR})

    sim = Composite(
        {
            "state": {
                "t": 0.0,
                "a": 0.0, "b": 0.0, "c": 0.0, "k": 0.0,
                "z": 0.0, "w": 0.0, "u": 0.0, "energy": 0.0,

                "tick": {
                    "_type": "process",
                    "address": TICK_ADDR,
                    "config": {},
                    "interval": dt,
                    "inputs": {"t": ["t"]},
                    "outputs": {"t": ["t"]},
                },

                "math": {
                    "_type": "step",
                    "address": MATH_ADDR,
                    "config": {
                        "expressions": [
                            {"out": "a", "expr": "2.0 + 1.2*sin(2*pi*0.35*t)"},
                            {"out": "b", "expr": "3.0 + 0.8*cos(2*pi*0.10*t + 0.4)"},
                            {"out": "c", "expr": "0.5 + 0.6*sin(2*pi*0.04*t)"},
                            {"out": "k", "expr": "1.0 + 2.0*(0.5 + 0.5*sin(2*pi*0.06*t))"},
                            {"out": "z", "expr": "a*b + sin(c)"},
                            {"out": "w", "expr": "z / (k + 1)"},
                            {"out": "u", "expr": "exp(-0.12*t) * cos(2*pi*0.55*t)"},
                            {"out": "energy", "expr": "0.5*(a**2 + b**2) + w**2 + 0.25*u**2"},
                        ],
                        "params": {"pi": float(np.pi)},
                        "functions": "numpy",
                        "debug": False,
                    },
                    "inputs": {"t": ["t"]},
                    "outputs": {
                        "a": ["a"], "b": ["b"], "c": ["c"], "k": ["k"],
                        "z": ["z"], "w": ["w"], "u": ["u"], "energy": ["energy"],
                    },
                },

                "emitter": emitter_from_wires(
                    {"t": ["t"], "a": ["a"], "b": ["b"], "c": ["c"], "k": ["k"],
                     "z": ["z"], "w": ["w"], "u": ["u"], "energy": ["energy"]}
                ),
            }
        },
        core=core,
    )

    sim.run(total_time)
    records = sim.state["emitter"]["instance"].query()

    print("n records:", len(records))
    print("first record:", records[0])
    print("last record :", records[-1])

    plot_timeseries(records, ["a", "b", "k"], title="Time-driven signals: a(t), b(t), k(t)")
    plot_timeseries(records, ["z"], title="z(t) = a(t)*b(t) + sin(c(t))", legend=False)
    plot_timeseries(records, ["w"], title="w(t) = z(t)/(k(t)+1)", legend=False)
    plot_timeseries(records, ["u"], title="u(t) damped oscillator", legend=False)
    plot_timeseries(records, ["energy"], title="energy(t)", legend=False)


def run_math_step_with_dependencies(core, total_time=8.0, dt=0.2):
    MATH_ADDR = f"local:!{MathExpressionStep.__module__}.MathExpressionStep"
    TICK_ADDR = f"local:!{Tick.__module__}.Tick"
    print("Using addresses:", {"tick": TICK_ADDR, "math": MATH_ADDR})

    sim = Composite(
        {"state": {
            "t": 0.0,
            "a": 0.0, "b": 0.0, "k": 0.0,
            "pre": 0.0, "z": 0.0, "w": 0.0, "rect": 0.0, "score": 0.0,

            "tick": {
                "_type": "process",
                "address": TICK_ADDR,
                "interval": dt,
                "inputs": {"t": ["t"]},
                "outputs": {"t": ["t"]},
            },

            "math": {
                "_type": "step",
                "address": MATH_ADDR,
                "config": {
                    "expressions": [
                        {"out": "score", "expr": "0.7*w + 0.3*rect - 0.15*k"},
                        {"out": "rect",  "expr": "log(1 + pre**2)"},
                        {"out": "w",     "expr": "tanh(z) / (1 + 0.3*k)"},
                        {"out": "z",     "expr": "0.8*pre + 0.6*sin(2*pi*0.07*t)"},
                        {"out": "pre",   "expr": "a*b"},
                        {"out": "k",     "expr": "1.5 + 1.0*sin(2*pi*0.03*t + 0.2)"},
                        {"out": "b",     "expr": "2.0*cos(2*pi*0.11*t) + 0.4*sin(2*pi*0.02*t)"},
                        {"out": "a",     "expr": "1.8*sin(2*pi*0.17*t) + 0.6*cos(2*pi*0.05*t + 0.3)"},
                    ],
                    "params": {"pi": float(np.pi)},
                    "functions": "numpy",
                    "debug": False,
                },
                "inputs": {"t": ["t"]},
                "outputs": {
                    "a": ["a"], "b": ["b"], "k": ["k"],
                    "pre": ["pre"], "z": ["z"], "w": ["w"],
                    "rect": ["rect"], "score": ["score"],
                },
            },

            "emitter": emitter_from_wires({
                "t": ["t"], "a": ["a"], "b": ["b"], "k": ["k"],
                "pre": ["pre"], "z": ["z"], "w": ["w"], "rect": ["rect"], "score": ["score"],
            }),
        }},
        core=core
    )

    sim.run(total_time)
    records = sim.state["emitter"]["instance"].query()

    print("n records:", len(records))
    print("first:", records[0])
    print("last :", records[-1])

    plot_timeseries(records, ["a", "b", "k"], title="Generated signals (scrambled list; dependency-ordered execution)")
    plot_timeseries(records, ["pre"], title="Intermediate: pre(t) = a*b", legend=False)
    plot_timeseries(records, ["z", "w"], title="z(t) and w(t)", legend=True)
    plot_timeseries(records, ["rect", "score"], title="rectification + score", legend=True)



if __name__ == "__main__":
    core = make_test_core()
    run_math_step_tick_only(core)
    run_math_step_with_dependencies(core)
