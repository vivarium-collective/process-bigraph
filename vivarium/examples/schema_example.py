"""
Use case to exercise types/schema:
 - Define a cell interface as outward pointing schema.
 - Find a cell type within a larger composite
 - Swap out a piece of the cell (chromosome type)?
 - Structure that can be interpreted in two different ways – one thing treats it as a neighboring cell/signaling companion, the other (a virus type) treats it as a host.
 - Automatic, simplified composition – one process (transcription) declares that it needs a chromosome to sample into, and the “matching” will automatically find the chromosome and plug into it.
 - Have a schema (chromosome), and then have the transcription process just declare chromosome for one of its ports.

functors examples:
 - a cell is in an environment. the cell's external state is a (type) float, the environment's state is a (type) 2D array of floats and dimensions, there is
 also a location that determines which of the env floats connects to the cell. Previously, this connection was mediated by an adaptor step. Can
 it be done entirely through the connection? This will let users more simply plug cells into an environment without having to worry about the adaptor
 step.
"""

import numpy as np
from vivarium.core.schema import Schema
from typing import List


# define some schema
class Concentration(Schema):

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'units': units.conc,
            'concentration': float
        }


class ExchangeSchema(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            m: Concentration() for m in self.config}


class LocationSchema(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'location': np.array
        }

# ListSchema demonstrates a way to configure a schema.
# This should be done more automatically in the Schema base class
# insight: the prior glob (*) schema can be just a DictSchema -- a parameterized type
class ListSchema(Schema):
    def __init__(self, config=None):
        super().__init__(config)
        self.element = self.config['parameters']['element']

    def validate(self, state):
        # check the state conforms to the schema
        schema = self.schema()
        if isinstance(state, schema['list']['_type']):
            for element in state:
                if not isinstance(element, schema['list']['element']):
                    return False
        else:
            return False
        return True

    def schema(self):
        return {
            'list': {
                '_type': list,
                'element': self.element
            }
        }

class CellSchema(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'exchange': Exchange(self.config['metabolites']),
            'location': LocationSchema(self.config['location'])
        }

class EnvironmentSchema(Schema):
    def __init__(self, config=None):
        super().__init__(config)
    def schema(self):
        return {
            'fields': np.array,
            'locations': ListSchema(element=LocationSchema()),  # can we use function keywords
        }

# define some processes
class Environment

# define a composite
class CellEnvironment(Composite):

class OtherChromosome(Composite):




def find_cell_schema():
    # make an environment with a cell in it
    cell_env = CellEnvironment()

    # find a composite that matches the cell schema
    cells = cell_env.query(CellSchema())

    # replace all the cells' chromosomes
    for cell in cells:
        cell.swap('chromosome', OtherChromosome())


def plug_cell_in_env():
    environment


if __name__ == '__main__':
    find_cell_schema()