from process_bigraph.processes.parameter_scan import ToySystem, ODE, RunProcess, ParameterScan
from process_bigraph.experiments.growth_division import Grow, Divide
from process_bigraph.experiments.minimal_gillespie import GillespieInterval, GillespieEvent


def register_processes(core):
    core.register_process('ToySystem', ToySystem)
    core.register_process('ToyODE', ODE)
    core.register_process('RunProcess', RunProcess)
    core.register_process('ParameterScan', ParameterScan)

    core.register_process('grow', Grow)
    core.register_process('divide', Divide)

    core.register_process('GillespieInterval', GillespieInterval)
    core.register_process('GillespieEvent', GillespieEvent)

    return core
