import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cobra
from cobra.io import load_model

from process_bigraph import Process, ProcessTypes, Composite
from process_bigraph.experiments.parameter_scan import RunProcess


# create new types
def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


# TODO -- check the function signature of the apply method and report missing keys upon registration

MODEL_FOR_TESTING = load_model('textbook')


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
        'model': 'Any',
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

        if self.config['model_file'] == 'TESTING':
            self.model = MODEL_FOR_TESTING
        elif not 'xml' in self.config['model_file']:
            # use the textbook model if no model file is provided
            self.model = load_model(self.config['model_file'])
        elif isinstance(self.config['model_file'], str):
            self.model = cobra.io.read_sbml_model(self.config['model_file'])
        else:
            # error handling
            raise ValueError('Invalid model file')

        for reaction_id, bounds in self.config['bounds'].items():
            if bounds['lower'] is not None:
                self.model.reactions.get_by_id(reaction_id).lower_bound = bounds['lower']
            if bounds['upper'] is not None:
                self.model.reactions.get_by_id(reaction_id).upper_bound = bounds['upper']

    def inputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

    def outputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

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

# Laplacian for 2D diffusion
LAPLACIAN_2D = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])


class DiffusionAdvection(Process):
    config_schema = {
        'n_bins': 'tuple[integer,integer]',
        'bounds': 'tuple[float,float]',
        'default_diffusion_rate': {'_type': 'float', '_default': 1e-1},
        'default_diffusion_dt': {'_type': 'float', '_default': 1e-1},
        'diffusion_coeffs': 'map[float]',
        'advection_coeffs': 'map[tuple[float,float]]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)

        # get diffusion rates
        bins_x = self.config['n_bins'][0]
        bins_y = self.config['n_bins'][1]
        length_x = self.config['bounds'][0]
        length_y = self.config['bounds'][1]
        dx = length_x / bins_x
        dy = length_y / bins_y
        dx2 = dx * dy

        # general diffusion rate
        diffusion_rate = self.config['default_diffusion_rate']
        self.diffusion_rate = diffusion_rate / dx2

        # diffusion rates for each individual molecules
        self.molecule_specific_diffusion = {
            mol_id: diff_rate / dx2
            for mol_id, diff_rate in self.config['diffusion_coeffs'].items()}

        # get diffusion timestep
        diffusion_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * diffusion_rate * (dx ** 2 + dy ** 2))
        self.diffusion_dt = min(diffusion_dt, self.config['default_diffusion_dt'])

    def inputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def outputs(self):
        return {
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'positive_float'
                },
            }
        }

    def update(self, state, interval):
        fields = state['fields']

        fields_update = {}
        for species, field in fields.items():
            fields_update[species] = self.diffusion_delta(
                field,
                interval,
                diffusion_coeff=self.config['diffusion_coeffs'][species],
                advection_coeff=self.config['advection_coeffs'][species]
            )

        return {
            'fields': fields_update
        }

    def diffusion_delta(self, state, interval, diffusion_coeff, advection_coeff):
        t = 0.0
        dt = min(interval, self.diffusion_dt)
        updated_state = state.copy()

        while t < interval:

            # Diffusion
            laplacian = convolve(
                updated_state,
                LAPLACIAN_2D,
                mode='reflect',
            ) * diffusion_coeff

            # Advection
            advective_flux_x = convolve(
                updated_state,
                np.array([[-1, 0, 1]]),
                mode='reflect',
            ) * advection_coeff[0]
            advective_flux_y = convolve(
                updated_state,
                np.array([[-1], [0], [1]]),
                mode='reflect',
            ) * advection_coeff[1]

            # Update the current state
            updated_state += (laplacian + advective_flux_x + advective_flux_y) * dt

            # # Ensure non-negativity
            # current_states[species] = np.maximum(updated_state, 0)

            # Update time
            t += dt

        return updated_state - state

def dfba_config(
        model_file='textbook',
        kinetic_params={
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)},
        biomass_reaction='Biomass_Ecoli_core',
        substrate_update_reactions={
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'},
        biomass_identifier='biomass',
        bounds={
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}}
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


def register_types(core):
    core.register('positive_float', {
        '_type': 'positive_float',
        '_inherit': 'float',
        '_apply': apply_non_negative})

    core.register('bounds', {
        'lower': 'maybe[float]',
        'upper': 'maybe[float]'})

    core.register_process(
        'DynamicFBA',
        DynamicFBA)

    core.register_process(
        'DiffusionAdvection',
        DiffusionAdvection)

    return core


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
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j]
                    }
                }
            }

    composite_state = {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'spatial_dfba': dfba_processes_dict
    }

    sim = Composite({'state': composite_state}, core=core)


    sim.update({}, 10.0)


def run_diffusion_process():
    n_bins = (4, 4)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    composite_state = {
        'fields': {
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': (10, 10),
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': {
                    'glucose': 1e-1,
                    'acetate': 1e-1,
                    'biomass': 1e-1,
                },
                'advection_coeffs': {
                    'glucose': (0, 0),
                    'acetate': (0, 0),
                    'biomass': (0, 0),
                },
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        }
    }

    sim = Composite({'state': composite_state}, core=core)
    # sim.add_emitter()

    sim.update({}, 10.0)

    data = sim.gather_results()

    print(data)


def run_comets(core):
    n_bins = (6, 6)

    initial_glucose = np.random.uniform(low=0, high=20, size=n_bins)
    initial_acetate = np.random.uniform(low=0, high=0, size=n_bins)
    initial_biomass = np.random.uniform(low=0, high=0.1, size=n_bins)

    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f'[{i},{j}]'] = {
                '_type': 'process',
                'address': 'local:DynamicFBA',
                'config': dfba_config(
                    model_file='TESTING'  # load the same model for all processes
                ),
                'inputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'glucose': ['..', 'fields', 'glucose', i, j],
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'biomass': ['..', 'fields', 'biomass', i, j]
                    }
                }
            }

    composite_state = {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            'glucose': initial_glucose,
            'acetate': initial_acetate,
            'biomass': initial_biomass,
        },
        'spatial_dfba': dfba_processes_dict,
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': n_bins,
                'bounds': (10, 10),
                'default_diffusion_rate': 1e-1,
                'default_diffusion_dt': 1e-1,
                'diffusion_coeffs': {
                    'glucose': 1e-1,
                    'acetate': 1e-1,
                    'biomass': 1e-1,
                },
                'advection_coeffs': {
                    'glucose': (0, 0),
                    'acetate': (0, 0),
                    'biomass': (0, 0),
                },
            },
            'inputs': {
                'fields': ['fields']
            },
            'outputs': {
                'fields': ['fields']
            }
        },
    }

    sim = Composite({
        'state': composite_state,
        'emitter': {
            'mode': 'all'}}, core=core)

    outdir = Path('out')
    filename = 'comets.json'

    # save the document
    sim.save(
        filename=filename,
        outdir=outdir,
        include_schema=True)

    sim.update({}, 100.0)

    results = sim.gather_results()

    load = Composite.load(
        path=outdir/filename,
        core=core)

    load.update({}, 100.0)

    other_results = load.gather_results()

    np.testing.assert_equal(
        results[('emitter',)][-1]['fields'],
        other_results[('emitter',)][-1]['fields'])

    print(results)



if __name__ == '__main__':
    core = ProcessTypes()
    core = register_types(core)

    # run_dfba_spatial(core)
    # run_diffusion_process(core)
    run_comets(core)
