from smoldyn import Simulation
from process_bigraph.emitter import DatabaseEmitter


model_fp = 'process_bigraph/experiments/model.txt'

sim = Simulation.fromFile(model_fp)

sim.addOutputData('time')
sim.addOutputData('mols')
sim.addCommand(cmd='executiontime time', cmd_type='E')
sim.addCommand(cmd='listmols mols', cmd_type='E')
sim.run(1, 0.5)
outputs = {
    'times': sim.getOutputData('time'),
    'molecules': sim.getOutputData('mols')
}

emitter = DatabaseEmitter(config={})
