"""Test the database emitter class with an instance used to emit and query simulation data."""


import os

from process_bigraph.emitter import DatabaseEmitter


def run_simulation(time):
    return [n/n**n for n in range(time)]


emitter_config = {
    'ports': {},
    'experiment_id': 'simulation_run_1',
    'emit_limit': 9000000,
}


emitter = DatabaseEmitter(emitter_config)


sim_runs = {
    str(n): run_simulation(n + n)
    for n in range(10)
}

sim_runs['table'] = emitter.experiment_id
sim_runs['data'] = {
    'time': 0,
    'values': {
        'output': 23
    }
}
emitter.emit(data=sim_runs)

print(emitter.db.phylogeny)

