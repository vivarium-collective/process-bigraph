import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
from bigraph_schema import get_path, set_path
from process_bigraph.composite import Step, Process


class Emitter(Step):
    """Base emitter class. An `Emitter` implementation instance diverts all querying of data to
        the primary historical collection whose type pertains to Emitter child, i.e:
            database-emitter=>`pymongo.Collection`, ram-emitter=>`.RamEmitter.history`(`List`)
    """
    config_schema = {
        'emit': 'schema'}

    def inputs(self) -> Dict:
        return self.config['emit']

    def query(self, query=None):
        return {}

    def update(self, state) -> Dict:
        return {}


class ConsoleEmitter(Emitter):

    def update(self, state) -> Dict:
        print(state)
        return {}


class RAMEmitter(Emitter):

    def __init__(self, config):
        super().__init__(config)
        self.history = []


    def update(self, state) -> Dict:
        self.history.append(copy.deepcopy(state))
        return {}


    def query(self, query=None):
        if isinstance(query, list):
            result = {}
            for path in query:
                element = get_path(self.history, path)
                result = set_path(result, path, element)
        else:
            result = self.history

        return result
