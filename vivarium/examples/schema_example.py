"""
Use case to exercise types/schema:
 - Define a cell interface as outward pointing schema.
 - Find a cell type within a larger composite
 - Swap out a piece of the cell (chromosome type)?
 - Structure that can be interpreted in two different ways – one thing treats it as a neighboring cell/signaling companion, the other (a virus type) treats it as a host.
 - Automatic, simplified composition – one process (transcription) declares that it needs a chromosome to sample into, and the “matching” will automatically find the chromosome and plug into it.
 - Have a schema (chromosome), and then have the transcription process just declare chromosome for one of its ports.

"""

import numpy as np
from vivarium.core.schema import Schema


# define some schema
class Concentration(Schema):

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self, config):
        return {
            'units': units.conc,
            'concentration': float
        }


class Exchange(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self, config):
        return {
            m: Concentration() for m in config}


class Location(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self, config):
        return {
            'location': np.array
        }


class Cell(Schema):
    def __init__(self, config=None):
        super().__init__(config)

    def schema(self, config):
        return {
            'exchange': Exchange(config['metabolites']),
            'location': Location(config['location'])
        }


# define a composite
class CellEnvironment(Composite):

class OtherChromosome(Composite):

    
def find_cell_schema():
    cell_env = CellEnvironment()

    cells = cell_env.query(Cell())

    for cell in cells:
        cell.swap('chromosome', OtherChromosome())


if __name__ == '__main__':
    find_cell_schema()