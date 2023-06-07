class ProcessProtocol():
    def __init__(self, config):
        pass

    def ports_schema(self):
        return {}

    def calculate_timestep(self, state):
        return 1.0

    def next_update(self, timestep, state):
        return {}


class LocalProtocol(ProcessProtocol):
    def __init__(self, location, config):
        self.process = lookup_process_somehow(location, config)

    def ports_schema(self):
        return self.process.ports_schema()

    def calculate_timestep(self, state):
        return self.process.calculate_timestep(state)

    def next_update(self, timestep, state):
        return self.process.next_update(timestep, state)


class HTTPProtocol(ProcessProtocol):
    def __init__(self, location, config):
        self.client = start_http_client(location, config)

    def ports_schema(self):
        schema = self.client.request('GET', {'method': 'ports_schema'})
        return schema
