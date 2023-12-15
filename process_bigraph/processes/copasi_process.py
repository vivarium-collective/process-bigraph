"""
Demo process for Copasi/Basico
"""


from process_bigraph import Process, Composite, process_registry, pf

try:
    from basico import (
        load_model,
        get_species,
        get_parameters,
        get_reactions,
        set_species,
        run_time_course,
        get_compartments,
        model_info
    )
except:
    raise ImportError('''
        You must install core-processes with the [copasi] optional requirement. 
        See the README for more information.
    ''')

class CopasiProcess(Process):
    config_schema = {'model_file': 'string'}

    def __init__(self, config=None):
        super().__init__(config)


        # Load the single cell model into Basico
        self.copasi_model_object = load_model(self.config['model_file'])

        # TODO -- make this configurable.
        self.input_ports = [
            'floating_species',
            # 'boundary_species',
            'model_parameters'
            # 'time',
            # 'compartments',
            # 'parameters',
            # 'stoichiometries',
        ]

        self.output_ports = [
            'floating_species',
            # 'time',
        ]

        # Get the species (floating and boundary)
        self.floating_species_list = get_species(model=self.copasi_model_object).index.tolist()
        self.floating_species_initial = get_species(model=self.copasi_model_object)['concentration'].tolist()
        # self.boundary_species_list = get_species(model=self.copasi_model_object).index.tolist()
        # self.boundary_species_initial = get_species(model=self.copasi_model_object)['concentration'].tolist()

        # Get the list of parameters and their values
        self.model_parameters_list = get_parameters(model=self.copasi_model_object).index.tolist()
        self.model_parameter_values = get_parameters(model=self.copasi_model_object)['initial_value'].tolist()

        # Get a list of reactions
        self.reaction_list = get_reactions(model=self.copasi_model_object).index.tolist()

        # Get a list of compartments
        self.compartments_list = get_compartments(model=self.copasi_model_object).index.tolist()

    def initial_state(self):
        floating_species_dict = dict(zip(self.floating_species_list, self.floating_species_initial))
        # boundary_species_dict = dict(zip(self.boundary_species_list, self.boundary_species_initial))
        model_parameters_dict = dict(zip(self.model_parameters_list, self.model_parameter_values))
        return {
            'time': 0.0,
            'floating_species': floating_species_dict,
            # 'boundary_species': boundary_species_dict,
            'model_parameters': model_parameters_dict
        }

    def schema(self):
        schema = {
            'time': 'float',
            'floating_species': {
                species_id: {
                    '_type': 'float',
                    '_apply': 'set',
                } for species_id in self.floating_species_list},   # TODO -- this should be a float with a set updater
            # 'boundary_species': {
            #     species_id: 'float' for species_id in self.boundary_species_list},
            'model_parameters': {
                param_id: 'float' for param_id in self.model_parameters_list},
            'reactions': {
                reaction_id: 'float' for reaction_id in self.reaction_list},
        }
        return schema

    def update(self, state, interval):

        # set copasi values according to what is passed in states
        for port_id, values in state.items():
            if port_id in self.input_ports:  # only update from input ports
                for cat_id, value in values.items():
                    set_species(name=cat_id, initial_concentration=value, model=self.copasi_model_object)

        # run model for "interval" length; we only want the state at the end
        timecourse = run_time_course(
            start_time=state['time'],
            duration=interval,
            intervals=1,
            update_model=True,
            model=self.copasi_model_object)

        # extract end values of concentrations from the model and set them in results
        results = {'time': interval}
        results['floating_species'] = {
            mol_id: float(get_species(name=mol_id, exact=True, model=self.copasi_model_object).concentration[0])
            for mol_id in self.floating_species_list}

        return results
