import pint
from process_bigraph.composite import Step

def UnitsAdapter(Step):
    config_schema = {
        'input_units': 'string',
        'output_units': 'string'}


    def initialize(self, config):
        self.input_units = units(self.config['input_units'])
        self.output_units = units(self.config['output_units'])


    def inputs(self):
        return self.config['input_units']


    def outputs(self):
        return self.config['output_units']


    def adapt(self, inputs):
        value = inputs * self.input_units
        return value.to(self.output_units)
