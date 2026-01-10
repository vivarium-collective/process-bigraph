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
    A Step that evaluates configured math expressions using SymPy -> NumPy.

    Config:
      expressions: list of {"out": <name>, "expr": <string>}
      params: dict of constants (not ports)
      functions: lambdify backend (default "numpy")
      debug: bool (default True) prints initialize-time diagnostics
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
        self._out_names = [spec["out"] for spec in expr_specs]
        if any((not isinstance(o, str) or not o) for o in self._out_names):
            raise ValueError("Each expression spec must have a non-empty string 'out'")
        if len(set(self._out_names)) != len(self._out_names):
            raise ValueError("Duplicate output names are not allowed")

        if debug:
            print("\n[MathExpressionStep] initialize()")
            print("  outputs:", self._out_names)
            print("  params :", self._params)
            print("  backend:", cfg.get("functions", "numpy"))

        # Parse expressions
        parsed = []
        if debug:
            print("  parsing expressions:")
        for i, spec in enumerate(expr_specs):
            out = spec["out"]
            expr_str = spec["expr"]
            if not isinstance(expr_str, str) or not expr_str:
                raise ValueError(f"Expression for '{out}' must be a non-empty string")

            try:
                expr = sp.sympify(expr_str)
            except Exception as e:
                raise ValueError(f"Failed to parse expression for '{out}': {expr_str}\n{e}") from e

            parsed.append((out, expr))

            if debug:
                free = sorted([str(s) for s in expr.free_symbols])
                print(f"    [{i}] {out} = {expr_str}")
                print(f"        sympy: {sp.srepr(expr)}")
                print(f"        free symbols: {free}")

        # Determine required input symbols in execution order
        param_syms = {sp.Symbol(k) for k in self._params.keys()}
        out_syms = {sp.Symbol(o) for o in self._out_names}

        required_inputs = set()
        available = set(param_syms)  # params available from start
        produced = set()

        if debug:
            print("  dependency scan (in order):")

        for out, expr in parsed:
            free = set(expr.free_symbols)

            future_outputs = out_syms - produced - {sp.Symbol(out)}
            if free & future_outputs:
                bad = ", ".join(sorted(str(s) for s in (free & future_outputs)))
                raise ValueError(
                    f"Expression for '{out}' references future output(s) before they exist: {bad}"
                )

            needed = free - available - produced
            required_inputs |= needed

            if debug:
                print(f"    out='{out}':")
                print(f"      produced so far : {[str(s) for s in sorted(produced, key=lambda s: str(s))]}")
                print(f"      params available: {[str(s) for s in sorted(param_syms, key=lambda s: str(s))]}")
                print(f"      needs as inputs : {[str(s) for s in sorted(needed, key=lambda s: str(s))]}")

            produced.add(sp.Symbol(out))
            available.add(sp.Symbol(out))

        self._in_symbols = sorted(required_inputs, key=lambda s: str(s))
        self._in_names = [str(s) for s in self._in_symbols]

        # Compile lambdas.
        self._param_symbols = [sp.Symbol(k) for k in self._params.keys()]
        self._out_symbols = [sp.Symbol(o) for o in self._out_names]
        self._arg_symbols = self._in_symbols + self._out_symbols + self._param_symbols

        backend = cfg.get("functions", "numpy")
        self._compiled = []
        for out, expr in parsed:
            fn = sp.lambdify(self._arg_symbols, expr, modules=backend)
            self._compiled.append((out, fn))

        if debug:
            print("  ✅ ports inferred")
            print("    inputs :", self._in_names)
            print("    outputs:", self._out_names)
            print("  ✅ lambdify arg order")
            print("    args   :", [str(s) for s in self._arg_symbols])
            print("  ✅ compiled expressions:", [o for o, _ in self._compiled])
            print("[MathExpressionStep] initialize complete\n")

        return cfg

    def inputs(self):
        return {name: "float" for name in self._in_names}

    def outputs(self):
        return {name: "overwrite[float]" for name in self._out_names}

    def update(self, state):
        values = {}

        for name in self._in_names:
            values[name] = float(state[name])

        for k, v in self._params.items():
            values[k] = float(v)

        out_patch = {}
        for out, fn in self._compiled:
            arg_vals = []
            for sym in self._arg_symbols:
                name = str(sym)
                if name in self._out_names and name not in values:
                    arg_vals.append(np.nan)
                else:
                    arg_vals.append(values.get(name, np.nan))

            y = fn(*arg_vals)
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


if __name__ == "__main__":
    core = make_test_core()
    run_math_step_tick_only(core)
