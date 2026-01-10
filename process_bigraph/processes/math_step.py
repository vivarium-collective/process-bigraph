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
import sympy as sp
import numpy as np
from process_bigraph.composite import Step


class MathExpressionStep(Step):
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

def run_math_step_tick_only(core, total_time=12.0, dt=0.1):
    """
    Interesting + intuitive demo with ONLY:
      - Tick (Process): advances time t
      - MathExpressionStep (Step): generates signals from time and combines them

    Story:
      - a(t): "daily activity"  (fast-ish sine)
      - b(t): "weekly mood"     (slower cosine)
      - c(t): "seasonal shift"  (very slow sine, used inside sin(c))
      - k(t): "knob" (0..~3) that modulates w = z/(k+1)
      - z(t): combined interaction: a*b + sin(c)
      - w(t): normalized z by a varying knob
      - u(t): damped oscillation (classic intuitive signal)
      - energy(t): a simple scalar summary
    """
    MATH_ADDR = f"local:!{MathExpressionStep.__module__}.MathExpressionStep"
    TICK_ADDR = f"local:!{Tick.__module__}.Tick"
    print("Using addresses:", {"tick": TICK_ADDR, "math": MATH_ADDR})

    sim = Composite(
        {
            "state": {
                # time
                "t": 0.0,

                # outputs (pre-init optional)
                "a": 0.0,
                "b": 0.0,
                "c": 0.0,
                "k": 0.0,
                "z": 0.0,
                "w": 0.0,
                "u": 0.0,
                "energy": 0.0,

                # Tick is the ONLY Process
                "tick": {
                    "_type": "process",
                    "address": TICK_ADDR,
                    "config": {},
                    "interval": dt,
                    "inputs": {"t": ["t"]},
                    "outputs": {"t": ["t"]},
                },

                # MathExpressionStep is the ONLY Step
                # It generates the "inputs" a,b,c,k from time t, then computes z,w,u,energy.
                "math": {
                    "_type": "step",
                    "address": MATH_ADDR,
                    "config": {
                        "expressions": [
                            # Time-driven signals (intuitive)
                            {"out": "a", "expr": "2.0 + 1.2*sin(2*pi*0.35*t)"},                # fast wave
                            {"out": "b", "expr": "3.0 + 0.8*cos(2*pi*0.10*t + 0.4)"},         # slower wave
                            {"out": "c", "expr": "0.5 + 0.6*sin(2*pi*0.04*t)"},                # slow drift
                            {"out": "k", "expr": "1.0 + 2.0*(0.5 + 0.5*sin(2*pi*0.06*t))"},    # knob in [1,3]

                            # Combine them
                            {"out": "z", "expr": "a*b + sin(c)"},                               # interaction + nonlinearity
                            {"out": "w", "expr": "z / (k + 1)"},                                # normalized by knob
                            {"out": "u", "expr": "exp(-0.12*t) * cos(2*pi*0.55*t)"},            # damped oscillator
                            {"out": "energy", "expr": "0.5*(a**2 + b**2) + w**2 + 0.25*u**2"},  # scalar summary
                        ],
                        # params are constants (not ports). Provide pi here.
                        "params": {"pi": float(np.pi)},
                        "functions": "numpy",
                    },
                    "inputs": {"t": ["t"]},
                    "outputs": {
                        "a": ["a"],
                        "b": ["b"],
                        "c": ["c"],
                        "k": ["k"],
                        "z": ["z"],
                        "w": ["w"],
                        "u": ["u"],
                        "energy": ["energy"],
                    },
                },

                # Record everything
                "emitter": emitter_from_wires(
                    {
                        "t": ["t"],
                        "a": ["a"],
                        "b": ["b"],
                        "c": ["c"],
                        "k": ["k"],
                        "z": ["z"],
                        "w": ["w"],
                        "u": ["u"],
                        "energy": ["energy"],
                    }
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

    # Plot a few key curves
    ts = np.array([r["t"] for r in records], dtype=float)
    z  = np.array([r["z"] for r in records], dtype=float)
    w  = np.array([r["w"] for r in records], dtype=float)
    u  = np.array([r["u"] for r in records], dtype=float)
    en = np.array([r["energy"] for r in records], dtype=float)
    a  = np.array([r["a"] for r in records], dtype=float)
    b  = np.array([r["b"] for r in records], dtype=float)
    k  = np.array([r["k"] for r in records], dtype=float)

    plt.figure()
    plt.plot(ts, z, marker=".")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title("z(t) = a(t)*b(t) + sin(c(t))")
    plt.show()

    plt.figure()
    plt.plot(ts, w, marker=".")
    plt.xlabel("t")
    plt.ylabel("w")
    plt.title("w(t) = z(t)/(k(t)+1)  (k is a time-varying knob)")
    plt.show()

    plt.figure()
    plt.plot(ts, u, marker=".")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("u(t) = exp(-0.12 t) * cos(2π·0.55 t)  (damped oscillation)")
    plt.show()

    plt.figure()
    plt.plot(ts, en, marker=".")
    plt.xlabel("t")
    plt.ylabel("energy")
    plt.title("energy(t) = 0.5(a²+b²) + w² + 0.25 u²")
    plt.show()

    plt.figure()
    plt.plot(ts, a, marker=".", label="a(t)")
    plt.plot(ts, b, marker=".", label="b(t)")
    plt.plot(ts, k, marker=".", label="k(t)")
    plt.xlabel("t")
    plt.ylabel("signals")
    plt.title("Time-driven signals generated by MathExpressionStep")
    plt.legend()
    plt.show()


def run_math_step_with_dependencies(core, total_time=8.0, dt=0.2):
    MATH_ADDR = f"local:!{MathExpressionStep.__module__}.MathExpressionStep"
    TICK_ADDR = f"local:!{Tick.__module__}.Tick"

    sim = Composite(
        {"state": {
            "t": 0.0,

            # outputs / intermediates (initialized but overwritten)
            "a": 0.0,
            "b": 0.0,
            "k": 0.0,
            "pre": 0.0,
            "z": 0.0,
            "w": 0.0,
            "rect": 0.0,
            "score": 0.0,

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
                    # SCRAMBLED ON PURPOSE (dependency sorter should fix it)
                    "expressions": [
                        # downstream first:
                        {"out": "score", "expr": "0.7*w + 0.3*rect - 0.15*k"},
                        {"out": "rect", "expr": "log(1 + pre**2)"},  # nonlinear rectification of pre
                        {"out": "w", "expr": "tanh(z) / (1 + 0.3*k)"},  # saturating + gain knob
                        {"out": "z", "expr": "0.8*pre + 0.6*sin(2*pi*0.07*t)"},  # mixes pre + slow modulation
                        {"out": "pre", "expr": "a*b"},  # interaction

                        # "inputs" generated from t:
                        {"out": "k", "expr": "1.5 + 1.0*sin(2*pi*0.03*t + 0.2)"},
                        {"out": "b", "expr": "2.0*cos(2*pi*0.11*t) + 0.4*sin(2*pi*0.02*t)"},
                        {"out": "a", "expr": "1.8*sin(2*pi*0.17*t) + 0.6*cos(2*pi*0.05*t + 0.3)"},
                    ],
                    "params": {"pi": float(np.pi)},
                    "functions": "numpy",
                    "debug": True,  # prints inferred order + deps
                },
                "inputs": {"t": ["t"]},
                "outputs": {
                    "a": ["a"],
                    "b": ["b"],
                    "k": ["k"],
                    "pre": ["pre"],
                    "z": ["z"],
                    "w": ["w"],
                    "rect": ["rect"],
                    "score": ["score"],
                },
            },

            "emitter": emitter_from_wires({
                "t": ["t"],
                "a": ["a"],
                "b": ["b"],
                "k": ["k"],
                "pre": ["pre"],
                "z": ["z"],
                "w": ["w"],
                "rect": ["rect"],
                "score": ["score"],
            }),
        }},
        core=core
    )

    sim.run(total_time)
    records = sim.state["emitter"]["instance"].query()

    print("n records:", len(records))
    print("first:", records[0])
    print("last :", records[-1])

    ts = np.array([r["t"] for r in records], float)
    a = np.array([r["a"] for r in records], float)
    b = np.array([r["b"] for r in records], float)
    k = np.array([r["k"] for r in records], float)
    pre = np.array([r["pre"] for r in records], float)
    z = np.array([r["z"] for r in records], float)
    w = np.array([r["w"] for r in records], float)
    rect = np.array([r["rect"] for r in records], float)
    score = np.array([r["score"] for r in records], float)

    plt.figure()
    plt.plot(ts, a, marker=".", label="a(t)")
    plt.plot(ts, b, marker=".", label="b(t)")
    plt.plot(ts, k, marker=".", label="k(t)")
    plt.xlabel("t");
    plt.ylabel("signals")
    plt.title("Generated signals (oscillatory, non-monotone)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ts, pre, marker=".")
    plt.xlabel("t");
    plt.ylabel("pre = a*b")
    plt.title("Intermediate interaction term pre(t)")
    plt.show()

    plt.figure()
    plt.plot(ts, z, marker=".", label="z")
    plt.plot(ts, w, marker=".", label="w")
    plt.xlabel("t");
    plt.ylabel("value")
    plt.title("z(t) and w(t): modulation + saturation + gain knob")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(ts, rect, marker=".", label="rect = log(1+pre^2)")
    plt.plot(ts, score, marker=".", label="score")
    plt.xlabel("t");
    plt.ylabel("value")
    plt.title("Nonlinear rectification + combined score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    core = make_test_core()
    # run_math_step_tick_only(core)
    run_math_step_with_dependencies(core)
