import abc
from bigraph_schema import fill, registry_registry, type_registry


class Process:
    config_schema = {}

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = fill(self.config_schema, config)

    @abc.abstractmethod
    def schema(self):
        return {}

    @abc.abstractmethod
    def apply(self, state, interval):
        return {}


class Engine(Process):
    def __init__(self, config=None):
        super().__init__(config)


class IncreaseProcess(Process):
    config_schema = {
        'rate': {
            '_type': 'float',
            '_default': '0.1',
        }
    }

    def __init__(self, config=None):
        super().__init__(config)

    def schema(self):
        return {
            'a': 'float',
        }
    
    def apply(self, state, interval):
        return {
            'a': state['a'] * self.config['rate']
        }


def test_process():
    process = IncreaseProcess()


if __name__ == '__main__':
    test_process()
