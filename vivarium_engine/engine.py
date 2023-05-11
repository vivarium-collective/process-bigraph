import abc
from bigraph_schema import fill, registry_registry, type_registry, apply_update


def hierarchy_depth(hierarchy, path=()):
    """
    Create a mapping of every path in the hierarchy to the node living at
    that path in the hierarchy.
    """

    base = {}

    for key, inner in hierarchy.items():
        down = tuple(path + (key,))
        if isinstance(inner, dict):
            base.update(hierarchy_depth(inner, down))
        else:
            base[down] = inner

    return base


def find_processes(state):
    found = {}

    for key, inner in state.items():
        if isinstance(inner, Process):
            found[key] = inner
        elif isinstance(inner, dict):
            result = find_processes(inner)
            if result:
                found[key] = result

    return found


def find_process_paths(state):
    processes = find_processes(state)
    return hierarchy_depth(processes)


def empty_front(t: float) -> Dict[str, Union[float, dict]]:
    return {
        'time': t,
        'update': {}
    }

# deal with steps vs temporal process vs edges

class Process:
    config_schema = {}

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config_schema.setdefault(
            'timestep',
            {'_type': 'float', '_default': '1.0'}
        )

        self.config = fill(
            self.config_schema,
            config
        )

    @abc.abstractmethod
    def schema(self):
        return {}

    def calculate_timestep(self, state):
        return self.config['timestep']

    @abc.abstractmethod
    def apply(self, state, interval):
        return {}


class Engine(Process):
    config_schema = {
        'schema': 'tree[any]',
        'instance': 'tree[any]',
        'initial_time': 'float',
        'global_time_precision': 'maybe[float]',
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.global_time = self.config['initial_time']
        self.state = fill(
            self.config['schema'],
            self.config['instance']
        )
        self.process_paths = find_process_paths(state)
        self.front: Dict = {
            path: empty_front(self.global_time)
            for path in self.process_paths}
        self.global_time_precision = self.config['global_time_precision']

    def schema(self):
        return self.config['schema']

    def run_process(self, path, process, )
        if path not in self.front:
            self.front[path] = empty_front(self.global_time)
        process_time = self.front[path]['time']
        if process_time <= self.global_time:
            if self.front[path].get('future'):
                future_front = self.front[path]['future']
                process_timestep = future_front['timestep']
                store = future_front['store']
                state = future_front['state']
                del self.front[path]['future']
            else:
                # get the time step
                store, state = self._process_state(path)
                process_timestep = process.calculate_timestep(state)

            if force_complete:
                # force the process to complete at end_time
                future = min(process_time + process_timestep, end_time)
            else:
                future = process_time + process_timestep
            if self.global_time_precision is not None:
                # set future time based on global_time_precision
                future = round(future, self.global_time_precision)

            if future <= end_time:

                # calculate the update for this process
                if process.update_condition(process_timestep, state):
                    update = self._process_update(
                        path,
                        process,
                        store,
                        state,
                        process_timestep)

                    # update front, to be applied at its projected time
                    self.front[path]['time'] = future
                    self.front[path]['update'] = update

                    # absolute timestep
                    timestep = future - self.global_time
                    if timestep < full_step:
                        full_step = timestep
                else:
                    # mark this path "quiet" so its time can be advanced
                    self.front[path]['update'] = (EmptyDefer(), store)
                    quiet_paths.append(path)
            else:
                # absolute timestep
                timestep = future - self.global_time
                if timestep < full_step:
                    full_step = timestep

        else:
            # don't shoot past processes that didn't run this time
            process_delay = process_time - self.global_time
            if process_delay < full_step:
                full_step = process_delay

        return full_step, quiet_paths
        

    def run(self, interval, force_complete=False):
        end_time = self.global_time + interval
        while self.global_time < end_time or force_complete:
            full_step = math.inf
            for path, process in self.process_paths.items():
            


    def apply(self, state, interval):
        # do everything
        


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
    process = IncreaseProcess({'rate': 0.2})
    schema = process.schema()
    state = fill(schema)
    update = process.apply({'a': 5.5}, 1.0)
    new_state = apply_update(schema, state, update)
    assert new_state['a'] == 1.1


def test_engine():
    


if __name__ == '__main__':
    test_process()
