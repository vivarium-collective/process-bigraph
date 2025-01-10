from process_bigraph.processes.parameter_scan import ToySystem, ODE, RunProcess, ParameterScan
from process_bigraph.processes.growth_division import Grow, Divide
# from process_bigraph.experiments.minimal_gillespie import GillespieInterval, GillespieEvent


TOY_PROCESSES = {
    'ToySystem': ToySystem,
    'ToyODE': ODE,
    'RunProcess': RunProcess,
    'ParameterScan': ParameterScan,
    'grow': Grow,
    'divide': Divide,
    # 'GillespieInterval': GillespieInterval,
    # 'GillespieEvent': GillespieEvent
}


def register_processes(core):
    for name, process in TOY_PROCESSES.items():
        core.register_process(name, process)
    return core
