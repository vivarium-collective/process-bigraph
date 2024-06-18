import numpy as np
from process_bigraph import Process, ProcessTypes, Composite
import matplotlib.pyplot as plt
import cobra
from cobra.io import load_model

core = ProcessTypes()


# create new types
def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)

positive_float = {
    '_type': 'positive_float',
    '_inherit': 'float',
    '_apply': apply_non_negative
}
core.register('positive_float', positive_float)

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'
}
core.register_process('bounds', bounds_type)


# TODO -- check the function signature of the apply method and report missing keys upon registration

class DynamicFBA(Process):
    """
    Performs dynamic FBA.

    Parameters:
    - model: The metabolic model for the simulation.
    - kinetic_params: Kinetic parameters (Km and Vmax) for each substrate.
    - biomass_reaction: The identifier for the biomass reaction in the model.
    - substrate_update_reactions: A dictionary mapping substrates to their update reactions.
    - biomass_identifier: The identifier for biomass in the current state.

    TODO -- check units
    """

    config_schema = {
        'model_file': 'string',
        'kinetic_params': 'map[tuple[float,float]]',
        'biomass_reaction': {
            '_type': 'string',
            '_default': 'Biomass_Ecoli_core'
        },
        'substrate_update_reactions': 'map[string]',
        'biomass_identifier': 'string',
        'bounds': 'map[bounds]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)

        if not 'xml' in self.config['model_file']:
            # use the textbook model if no model file is provided
            self.model = load_model(self.config['model_file'])
        else:
            self.model = cobra.io.read_sbml_model(self.config['model_file'])

        for reaction_id, bounds in self.config['bounds'].items():
            if bounds['lower'] is not None:
                self.model.reactions.get_by_id(reaction_id).lower_bound = bounds['lower']
            if bounds['upper'] is not None:
                self.model.reactions.get_by_id(reaction_id).upper_bound = bounds['upper']

    def inputs(self):
        return {'substrates': 'map[positive_float]'}

    def outputs(self):
        return {'substrates': 'map[positive_float]'}

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, state, interval):
        substrates_input = state['substrates']

        for substrate, reaction_id in self.config['substrate_update_reactions'].items():
            Km, Vmax = self.config['kinetic_params'][substrate]
            substrate_concentration = substrates_input[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        substrate_update = {}

        solution = self.model.optimize()
        if solution.status == 'optimal':
            current_biomass = substrates_input[self.config['biomass_identifier']]
            biomass_growth_rate = solution.fluxes[self.config['biomass_reaction']]
            substrate_update[self.config['biomass_identifier']] = biomass_growth_rate * current_biomass * interval

            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                flux = solution.fluxes[reaction_id]
                substrate_update[substrate] = flux * current_biomass * interval
                # TODO -- assert not negative?
        else:
            # Handle non-optimal solutions if necessary
            # print('Non-optimal solution, skipping update')
            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                substrate_update[substrate] = 0

        return {
            'substrates': substrate_update,
        }


core.register_process('DynamicFBA', DynamicFBA)

from process_bigraph.experiments.parameter_scan import RunProcess


def dfba_config(
        model_file='textbook',
        kinetic_params={'glucose': (0.5, 1), 'acetate': (0.5, 2)},
        biomass_reaction='Biomass_Ecoli_core',
        substrate_update_reactions={'glucose': 'EX_glc__D_e', 'acetate': 'EX_ac_e'},
        biomass_identifier='biomass',
        bounds={'EX_o2_e': {'lower': -2, 'upper': None}, 'ATPM': {'lower': 1, 'upper': 1}}
):
    return {
        'model_file': model_file,
        'kinetic_params': kinetic_params,
        'biomass_reaction': biomass_reaction,
        'substrate_update_reactions': substrate_update_reactions,
        'biomass_identifier': biomass_identifier,
        'bounds': bounds
    }


# TODO -- this should be imported, or just part of Process?
def run_process(
        address,
        config,
        core_type,
        initial_state,
        observables,
        timestep=1,
        runtime=10
):
    config = {
        'process_address': address,
        'process_config': config,
        'observables': observables,
        'timestep': timestep,
        'runtime': runtime}

    run = RunProcess(config, core_type)
    return run.update(initial_state)


def run_dfba_spatial():
    n_bins = (2, 2)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f'[{i},{j}]'] = {
                '_type': 'process',
                'address': 'local:DynamicFBA',
                'config': dfba_config(),
                'inputs': {
                    'substrates': {
                        'glucose': ['..', '..', 'fields', 'glucose', i, j],
                        'acetate': ['..', '..', 'fields', 'acetate', i, j],
                        'biomass': ['..', '..', 'fields', 'biomass', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'glucose': ['..', '..', 'fields', 'glucose', i, j],
                        'acetate': ['..', '..', 'fields', 'acetate', i, j],
                        'biomass': ['..', '..', 'fields', 'biomass', i, j]
                    }
                }
            }

    composite_state = {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_size': n_bins,
                '_value': 'positive_float'
            },
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'spatial_dfba': dfba_processes_dict
    }

    sim = Composite({'state': composite_state}, core=core)


if __name__ == '__main__':
    run_dfba_spatial()
