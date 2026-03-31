"""
Benchmark suite for process-bigraph runtime.

Tests runtime performance under varying conditions:
- Number of processes
- Number of ports per process
- State tree depth / communication overhead
- Different wiring patterns (local, cross-tree, many-to-one)

Usage:
    python benchmarks/benchmark_runtime.py
    python benchmarks/benchmark_runtime.py --profile    # with cProfile breakdown
"""

import sys
import os
import time
import json
import inspect
import cProfile
import pstats
import io
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

# Add project root to path so we can import properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigraph_schema import allocate_core
from process_bigraph.composite import Process, Step, Composite


# =====================
# Benchmark Processes
# =====================

class TrivialProcess(Process):
    """Minimal process — isolates communication overhead from computation."""
    config_schema = {
        'rate': {'_type': 'float', '_default': '0.01'},
    }

    def inputs(self):
        return {f'port_{i}': 'float' for i in range(self.config.get('n_ports', 1))}

    def outputs(self):
        return {f'port_{i}': 'float' for i in range(self.config.get('n_ports', 1))}

    def update(self, state, interval):
        return {k: v * self.config['rate'] for k, v in state.items()}


class HeavyPortProcess(Process):
    """Process with configurable number of ports."""
    config_schema = {
        'rate': {'_type': 'float', '_default': '0.01'},
        'n_ports': {'_type': 'integer', '_default': '1'},
    }

    def inputs(self):
        return {f'port_{i}': 'float' for i in range(self.config['n_ports'])}

    def outputs(self):
        return {f'port_{i}': 'float' for i in range(self.config['n_ports'])}

    def update(self, state, interval):
        return {k: v * self.config['rate'] for k, v in state.items()}


class ArrayPortProcess(Process):
    """Process that communicates arrays — tests serialization overhead."""
    config_schema = {
        'rate': {'_type': 'float', '_default': '0.01'},
        'n_ports': {'_type': 'integer', '_default': '1'},
    }

    def inputs(self):
        return {f'port_{i}': 'map[float]' for i in range(self.config['n_ports'])}

    def outputs(self):
        return {f'port_{i}': 'map[float]' for i in range(self.config['n_ports'])}

    def update(self, state, interval):
        result = {}
        for k, v in state.items():
            if isinstance(v, dict):
                result[k] = {mk: mv * self.config['rate'] for mk, mv in v.items()}
            else:
                result[k] = v
        return result


# =====================
# Composite Builders
# =====================

def make_core():
    """Create a core with benchmark processes registered."""
    members = dict(inspect.getmembers(sys.modules[__name__]))
    core = allocate_core(top=members)
    return core


def build_n_process_composite(n_processes, n_ports=1, interval=1.0, core=None):
    """Build a composite with n independent processes, each with n_ports ports."""
    core = core or make_core()

    state = {}
    # Create shared state variables
    for p in range(n_processes):
        for port in range(n_ports):
            state[f'var_{p}_{port}'] = 1.0

    # Create processes
    for p in range(n_processes):
        inputs = {f'port_{i}': [f'var_{p}_{i}'] for i in range(n_ports)}
        outputs = {f'port_{i}': [f'var_{p}_{i}'] for i in range(n_ports)}
        state[f'process_{p}'] = {
            '_type': 'process',
            'address': f'local:{HeavyPortProcess.__name__}',
            'config': {'rate': 0.01, 'n_ports': n_ports},
            'interval': interval,
            'inputs': inputs,
            'outputs': outputs,
        }

    composite = Composite({'state': state}, core=core)
    return composite


def build_shared_state_composite(n_processes, n_shared_vars=1, core=None):
    """Build a composite where multiple processes read/write the same variables.

    This stresses the communication layer since all processes contend on shared state.
    """
    core = core or make_core()

    state = {}
    for v in range(n_shared_vars):
        state[f'shared_{v}'] = 1.0

    for p in range(n_processes):
        inputs = {f'port_{i}': [f'shared_{i}'] for i in range(n_shared_vars)}
        outputs = {f'port_{i}': [f'shared_{i}'] for i in range(n_shared_vars)}
        state[f'process_{p}'] = {
            '_type': 'process',
            'address': f'local:{HeavyPortProcess.__name__}',
            'config': {'rate': 0.01, 'n_ports': n_shared_vars},
            'interval': 1.0,
            'inputs': inputs,
            'outputs': outputs,
        }

    composite = Composite({'state': state}, core=core)
    return composite


def build_nested_composite(depth, processes_per_level=2, n_ports=1, core=None):
    """Build a composite with nested sub-composites to test deep state trees."""
    core = core or make_core()

    def make_level_state(level, prefix=''):
        state = {}
        if level == 0:
            # Leaf level: actual processes
            for p in range(processes_per_level):
                name = f'{prefix}proc_{p}'
                var_name = f'{prefix}var_{p}'
                state[var_name] = 1.0
                inputs = {f'port_{i}': [var_name] for i in range(n_ports)}
                outputs = {f'port_{i}': [var_name] for i in range(n_ports)}
                state[name] = {
                    '_type': 'process',
                    'address': f'local:{HeavyPortProcess.__name__}',
                    'config': {'rate': 0.01, 'n_ports': n_ports},
                    'interval': 1.0,
                    'inputs': inputs,
                    'outputs': outputs,
                }
        else:
            # Intermediate level: create children
            for c in range(processes_per_level):
                child_prefix = f'{prefix}L{level}_C{c}_'
                child_state = make_level_state(level - 1, child_prefix)
                state.update(child_state)
        return state

    state = make_level_state(depth)
    composite = Composite({'state': state}, core=core)
    return composite


def build_map_state_composite(n_processes, map_size=10, core=None):
    """Build a composite where processes exchange map[float] data.

    Tests overhead from structured data communication.
    """
    core = core or make_core()

    state = {}
    for p in range(n_processes):
        state[f'map_var_{p}'] = {f'key_{k}': 1.0 for k in range(map_size)}

    for p in range(n_processes):
        state[f'process_{p}'] = {
            '_type': 'process',
            'address': f'local:{ArrayPortProcess.__name__}',
            'config': {'rate': 0.01, 'n_ports': 1},
            'interval': 1.0,
            'inputs': {'port_0': [f'map_var_{p}']},
            'outputs': {'port_0': [f'map_var_{p}']},
        }

    composite = Composite({'state': state}, core=core)
    return composite


# =====================
# Timing Utilities
# =====================

@dataclass
class BenchmarkResult:
    name: str
    n_processes: int
    n_ports: int
    n_timesteps: int
    total_time: float
    init_time: float
    run_time: float
    time_per_step: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self):
        return (
            f"  {self.name:40s} | "
            f"procs={self.n_processes:4d} | "
            f"ports={self.n_ports:3d} | "
            f"steps={self.n_timesteps:4d} | "
            f"init={self.init_time:8.4f}s | "
            f"run={self.run_time:8.4f}s | "
            f"per_step={self.time_per_step*1000:8.3f}ms"
        )


def time_composite_run(name, build_fn, build_kwargs, n_timesteps=10, core=None):
    """Time the initialization and run of a composite."""
    core = core or make_core()

    t0 = time.perf_counter()
    composite = build_fn(core=core, **build_kwargs)
    t1 = time.perf_counter()
    init_time = t1 - t0

    t2 = time.perf_counter()
    composite.run(float(n_timesteps))
    t3 = time.perf_counter()
    run_time = t3 - t2

    n_procs = build_kwargs.get('n_processes', 0)
    n_ports = build_kwargs.get('n_ports', 1)

    return BenchmarkResult(
        name=name,
        n_processes=n_procs,
        n_ports=n_ports,
        n_timesteps=n_timesteps,
        total_time=init_time + run_time,
        init_time=init_time,
        run_time=run_time,
        time_per_step=run_time / n_timesteps if n_timesteps > 0 else 0,
        extra=build_kwargs,
    )


def profile_composite_run(name, build_fn, build_kwargs, n_timesteps=10, core=None):
    """Profile a composite run with cProfile and return top functions."""
    core = core or make_core()
    composite = build_fn(core=core, **build_kwargs)

    profiler = cProfile.Profile()
    profiler.enable()
    composite.run(float(n_timesteps))
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    return stream.getvalue()


# =====================
# Benchmark Scenarios
# =====================

def run_scaling_benchmarks(core=None):
    """Test how runtime scales with number of processes."""
    core = core or make_core()
    results = []
    print("\n=== Scaling: Number of Processes (1 port each) ===")
    for n in [1, 2, 5, 10, 20, 50, 100]:
        r = time_composite_run(
            f"scale_processes_{n}",
            build_n_process_composite,
            {'n_processes': n, 'n_ports': 1},
            n_timesteps=10,
            core=core,
        )
        results.append(r)
        print(r.summary())
    return results


def run_port_scaling_benchmarks(core=None):
    """Test how runtime scales with number of ports per process."""
    core = core or make_core()
    results = []
    print("\n=== Scaling: Number of Ports (10 processes) ===")
    for n_ports in [1, 2, 5, 10, 20, 50]:
        r = time_composite_run(
            f"scale_ports_{n_ports}",
            build_n_process_composite,
            {'n_processes': 10, 'n_ports': n_ports},
            n_timesteps=10,
            core=core,
        )
        results.append(r)
        print(r.summary())
    return results


def run_shared_state_benchmarks(core=None):
    """Test contention: multiple processes on the same variables."""
    core = core or make_core()
    results = []
    print("\n=== Shared State Contention ===")
    for n_procs in [2, 5, 10, 20, 50]:
        r = time_composite_run(
            f"shared_state_{n_procs}procs",
            build_shared_state_composite,
            {'n_processes': n_procs, 'n_shared_vars': 3},
            n_timesteps=10,
            core=core,
        )
        results.append(r)
        print(r.summary())
    return results


def run_map_data_benchmarks(core=None):
    """Test structured data overhead with map[float] ports."""
    core = core or make_core()
    results = []
    print("\n=== Map Data Communication Overhead ===")
    for map_size in [1, 5, 10, 50, 100]:
        r = time_composite_run(
            f"map_data_size_{map_size}",
            build_map_state_composite,
            {'n_processes': 10, 'map_size': map_size},
            n_timesteps=10,
            core=core,
        )
        results.append(r)
        print(r.summary())
    return results


def run_timestep_scaling_benchmarks(core=None):
    """Test how runtime scales with number of timesteps."""
    core = core or make_core()
    results = []
    print("\n=== Scaling: Number of Timesteps (10 processes, 5 ports) ===")
    for n_steps in [1, 5, 10, 50, 100]:
        r = time_composite_run(
            f"scale_timesteps_{n_steps}",
            build_n_process_composite,
            {'n_processes': 10, 'n_ports': 5},
            n_timesteps=n_steps,
            core=core,
        )
        results.append(r)
        print(r.summary())
    return results


def run_all_benchmarks():
    """Run all benchmark scenarios."""
    core = make_core()
    all_results = []

    all_results.extend(run_scaling_benchmarks(core))
    all_results.extend(run_port_scaling_benchmarks(core))
    all_results.extend(run_shared_state_benchmarks(core))
    all_results.extend(run_map_data_benchmarks(core))
    all_results.extend(run_timestep_scaling_benchmarks(core))

    return all_results


def run_profile(scenario='default'):
    """Run cProfile on a representative scenario and print top functions."""
    core = make_core()

    scenarios = {
        'default': (build_n_process_composite, {'n_processes': 20, 'n_ports': 5}, 20),
        'many_processes': (build_n_process_composite, {'n_processes': 50, 'n_ports': 1}, 10),
        'many_ports': (build_n_process_composite, {'n_processes': 10, 'n_ports': 20}, 10),
        'shared_state': (build_shared_state_composite, {'n_processes': 20, 'n_shared_vars': 5}, 10),
        'map_data': (build_map_state_composite, {'n_processes': 10, 'map_size': 50}, 10),
    }

    if scenario not in scenarios:
        print(f"Unknown scenario '{scenario}'. Available: {list(scenarios.keys())}")
        return

    build_fn, kwargs, n_steps = scenarios[scenario]
    print(f"\n=== Profile: {scenario} ({n_steps} timesteps) ===")
    profile_output = profile_composite_run(
        scenario, build_fn, kwargs, n_timesteps=n_steps, core=core)
    print(profile_output)


# =====================
# Main
# =====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark process-bigraph runtime')
    parser.add_argument('--profile', action='store_true', help='Run cProfile on default scenario')
    parser.add_argument('--scenario', default='default',
                        help='Profile scenario: default, many_processes, many_ports, shared_state, map_data')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    if args.profile:
        run_profile(args.scenario)
    else:
        results = run_all_benchmarks()
        if args.json:
            print(json.dumps([asdict(r) for r in results], indent=2))

        # Print summary
        print("\n=== Summary ===")
        print(f"Total benchmarks run: {len(results)}")
        slowest = max(results, key=lambda r: r.time_per_step)
        fastest = min(results, key=lambda r: r.time_per_step)
        print(f"Fastest per-step: {fastest.name} ({fastest.time_per_step*1000:.3f}ms)")
        print(f"Slowest per-step: {slowest.name} ({slowest.time_per_step*1000:.3f}ms)")
