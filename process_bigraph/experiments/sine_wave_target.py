import sys
import inspect
import numpy as np

import matplotlib.pyplot as plt

from process_bigraph import allocate_core
from process_bigraph.composite import Process, Step, Composite, as_process, as_step
from process_bigraph.emitter import emitter_from_wires


def rebuild_core():
    top = dict(inspect.getmembers(sys.modules["__main__"]))
    return allocate_core(top=top)

core = rebuild_core()

@as_step(
    inputs={"a": "float", "b": "float"},
    outputs={"sum": "overwrite[float]"},
)
def update_add(state):
    return {"sum": float(state["a"]) + float(state["b"])}

# ---- Steps for the workflow ----
@as_step(
    inputs={"t": "float"},
    outputs={"target": "overwrite[float]"},
)
def update_sine_target(state):
    """
    target(t) = center + amplitude * sin(omega * t)
    """
    t = float(state["t"])
    center = 5.0
    amplitude = 5.0
    period = 8.0
    omega = 2.0 * np.pi / period

    print(f'state {state}, target: {center + amplitude * np.sin(omega * t)}')
    return {"target": center + amplitude * np.sin(omega * t)}

@as_step(
    inputs={"x": "float", "target": "float"},
    outputs={"error": "overwrite[float]"},
)
def update_error(state):
    return {"error": float(state["target"]) - float(state["x"])}




class MoveToward(Process):
    """
    Move x toward target at speed 'rate' per unit time.
    (stateful Process: gets an interval argument)
    """
    config_schema = {
        "rate": {
            "_type": "float",
            "_default": 1.0,
        }
    }

    def initialize(self, config=None):
        if self.config["rate"] < 0:
            raise ValueError("MoveToward requires rate >= 0")

    def inputs(self):
        return {"x": "float", "target": "float"}

    def outputs(self):
        return {"x": "float"}

    def update(self, state, interval):
        x = float(state["x"])
        target = float(state["target"])
        rate = float(self.config["rate"])

        # bounded movement toward target
        max_step = rate * interval
        delta = np.clip(target - x, -max_step, max_step)
        return {"x": delta}

# Resolve address for the notebook-defined Process (recommended in notebooks)
MT_ADDR = f"local:!{MoveToward.__module__}.MoveToward"
print("Using MoveToward address:", MT_ADDR)

# Quick sanity check: instantiate + run once
p = MoveToward(config={"rate": 2.0}, core=core)
print("update:", p.update({"x": 0.0, "target": 10.0}, interval=1.0))  # expect x=2.0
print("✅ MoveToward Process defined")



def run_1():
    # ---- Initial conditions ----
    initial_env = {
        "t": 0.0,
        "x": 0.0,
        "target": 0.0,
        "error": 0.0,
    }

    workflow_with_emitter = Composite(
        {
            "state": {
                "Env": {
                    "t": 0.0,
                    "x": 0.0,
                    "target": 0.0,
                    "error": 0.0,
                },
                # Step: compute moving target from global time
                "targeter": {
                    "_type": "step",
                    "address": "local:sine_target",
                    "inputs": {"t": ["global_time"]},
                    "outputs": {"target": ["Env", "target"]},
                },

                # Process: move x toward the (changing) target
                "mover": {
                    "_type": "process",
                    "address": MT_ADDR,
                    "config": {"rate": 1.0},
                    "interval": 1,
                    "inputs": {
                        "x": ["Env", "x"],
                        "target": ["Env", "target"],
                    },
                    "outputs": {
                        "x": ["Env", "x"],
                    },
                },

                # # Step: compute error after mover updates x
                # "observer": {
                #     "_type": "step",
                #     "address": "local:error",
                #     "inputs": {
                #         "x": ["Env", "x"],
                #         "target": ["Env", "target"],
                #     },
                #     "outputs": {
                #         "error": ["Env", "error"],
                #     },
                # },

                # Emitter: record specified wires each tick
                "emitter": emitter_from_wires(
                    {
                        "time": ["global_time"],
                        "t": ["Env", "t"],
                        "x": ["Env", "x"],
                        "target": ["Env", "target"],
                        "error": ["Env", "error"],
                    }
                ),
            }
        },
        core=core,
    )

    # Run and then read back recorded rows
    workflow_with_emitter.run(10.0)

    records = workflow_with_emitter.state["emitter"]["instance"].query()
    print("n records:", len(records))
    print("first record:", records[0])
    print("last record:", records[-1])

    records  # in notebooks, this will display nicely


    # Convert records (list of dicts) to columns
    times = [r["time"] for r in records]
    x_vals = [r["x"] for r in records]
    target_vals = [r["target"] for r in records]
    error_vals = [r["error"] for r in records]

    plt.figure(figsize=(8, 4))

    plt.plot(times, x_vals, label="x (state)", linewidth=2)
    plt.plot(times, target_vals, label="target", linestyle="--")
    plt.plot(times, error_vals, label="error", linestyle=":")

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("MoveToward workflow with time-varying target")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_1()
