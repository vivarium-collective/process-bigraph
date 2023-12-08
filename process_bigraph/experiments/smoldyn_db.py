from smoldyn import Simulation
from process_bigraph.emitter.utils import DatabaseEmitter


model_fp = 'process_bigraph/experiments/crowding-model.txt'

sim = Simulation.fromFile(model_fp)

sim.addOutputData('time')
sim.addOutputData('mols')
sim.addCommand(cmd='executiontime time', cmd_type='E')
sim.addCommand(cmd='listmols mols', cmd_type='E')
sim.run(1, 1)

molecules = sim.getOutputData('mols')
time = sim.getOutputData('time')

emitter = DatabaseEmitter(config={'ports': {}, 'emit_limit': 9000000})
data = emitter.format_data(table_id='minE_0', time='1', mol_list=molecules)

emitter.emit(data)



