"""Test the database emitter class with an instance used to emit and query simulation data."""

from process_bigraph.composite import ProcessTypes
from process_bigraph.emitter import DatabaseEmitter


def run_simulation(time):
    return [n/n**n for n in range(time)]


def test_database_emitter():
    core = ProcessTypes()

    emitter_config = {
        'ports': {},
        'experiment_id': 'simulation_run_1',
        'emit_limit': 9000000}

    emitter = DatabaseEmitter(emitter_config, core)

    sim_runs = {
        'data': {}}

    sim_runs['table'] = emitter.experiment_id

    for t in range(10):
        sim_runs['data'][str(t)] = {
            'time': t,
            'values': {
                'output': run_simulation(t)}}

    print(sim_runs.get('data'))

    emitter.update(sim_runs)

    result = emitter.query([('data', 'values', 'output')])


