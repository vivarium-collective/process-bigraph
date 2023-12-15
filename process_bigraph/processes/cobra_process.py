"""
COBRA FBA Process
"""


from process_bigraph import Process, Step, Composite, process_registry


try:
    from cobra.io import read_sbml_model
    # from cobra_process.library import pf
except:
    raise ImportError('''
        You must install core-processes with the [cobra] optional requirement. 
        See the README for more information.
    ''')


class CobraProcess(Process):
    config_schema = {
        'model_file': 'string'
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model = read_sbml_model(self.config['model_file'])
        self.reactions = self.model.reactions
        self.metabolites = self.model.metabolites
        self.objective = self.model.objective
        self.boundary = self.model.boundary

    def initial_state(self):
        solution = self.model.optimize()
        optimized_fluxes = solution.fluxes

        state = {'fluxes': {}, 'reaction_bounds': {}}
        for reaction in self.model.reactions:
            state['fluxes'][reaction.id] = optimized_fluxes[reaction.id]
            state['reaction_bounds'][reaction.id] = {
                'lower_bound': reaction.lower_bound,
                'upper_bound': reaction.upper_bound}
        return state

    def schema(self):
        return {
            'fluxes': {
                reaction.id: 'float' for reaction in self.reactions
            },
            'objective_value': 'float',
            'reaction_bounds': {
                reaction.id: {
                    'upper_bound': 'float',
                    'lower_bound': 'float'
                } for reaction in self.reactions
            },
        }

    def update(self, state, interval):

        # set reaction bounds
        reaction_bounds = state['reaction_bounds']
        for reaction_id, bounds in reaction_bounds.items():
            self.model.reactions.get_by_id(reaction_id).bounds = (bounds['lower_bound'], bounds['upper_bound'])

        # run solver
        solution = self.model.optimize()

        return {
            'fluxes': solution.fluxes.to_dict(),
            'objective_value': solution.objective_value
        }
