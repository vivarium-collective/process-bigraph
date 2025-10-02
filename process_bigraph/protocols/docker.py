"""
===============================================
Protocol for running processes in parallel using
python multiprocessing
===============================================
"""

import sys
import json
import time
import uuid
import pstats
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple

import docker

from docker import DockerClient

from process_bigraph.composite import Process, SyncUpdate
from process_bigraph.protocols.protocol import Protocol
from process_bigraph.protocols.local import LocalProtocol


def setup_working_directory(path):
    if not path.exists():
        path.mkdir(
            parents=True,
            exist_ok=True)


DOCKER_WORKING_PATH = Path('containers')


def receive(sock, buffer_size=4096):
    data = bytearray()
    while True:
        packet = sock.recv(buffer_size)
        data.extend(packet)
        if packet[-1] == 10:
            break
    return data


class DockerProcess(Process):
    def __init__(self, client, data, config, core) -> None:
        """
        Sets up a docker container to be a process and communicates with
        it over a socket.
        """

        self._ended = False
        self._pending_command: Optional[
            Tuple[str, Optional[tuple], Optional[dict]]] = None

        self.client = client
        self.data = data
        self.image = self.data['image']
        self.host = self.data.get('host', '0.0.0.0')
        self.port = self.data.get('port', 11111)

        self.uuid = str(uuid.uuid4())
        self.work_path = DOCKER_WORKING_PATH / self.uuid
        self.config_path = self.work_path / 'config'

        setup_working_directory(self.config_path)

        with open(self.config_path / 'config.json', 'w') as config_file:
            json.dump(config, config_file)

        self.container = self.client.containers.run(
            self.image,
            ports={f'{self.port}/tcp': str(self.port)},
            volumes={
                str(self.config_path.absolute()): {
                    'bind': '/config', 'mode': 'ro'}},
            detach=True)

        # wait to start the socket until the container is running (!)
        while self.container.status != "running":
            print(f"'{self.image}' status: {self.container.status} - waiting...")
            time.sleep(1)
            self.container.reload()

        print(f'{self.image} now running')

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM)

        self.socket.connect((
            self.host,
            self.port))

        super().__init__(config, core=core)

    def send_command(
            self, command: str, args: Optional[tuple] = None,
            kwargs: Optional[dict] = None,
            run_pre_check: bool = True) -> None:
        '''Send a command to the parallel process.

        See :py:func:``_handle_parallel_process`` for details on how the
        command will be handled.
        '''
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)

        send = {
            'command': command,
            'arguments': {}}

        if command == 'update':
            state, interval = args
            send['arguments'] = {
                'state': state,
                'interval': interval}

        send_json = f'{json.dumps(send)}\n'.encode('utf-8')

        print(f'sending {send_json}')

        self.socket.sendall(
            send_json)

    def get_command_result(self):
        """Get the result of a command sent to the parallel process.

        Commands and their results work like a queue, so unlike
        :py:class:`Process`, you can technically call this method
        multiple times and get different return values each time.
        This behavior is subject to change, so you should not rely on
        it.

        Returns:
            The command result.
        """

        if not self._pending_command:
            raise RuntimeError(
                'Trying to retrieve command result, but no command is '
                'pending.')
        self._pending_command = None

        result_json = receive(self.socket)
        result = json.loads(result_json)

        return result

    def get(self):
        return self.get_command_result()

    @staticmethod
    def generate_state(config=None):
        """Generate static initial state for user configuration or inspection."""
        return {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, config):
        self._config = config
        
    @property
    def composition(self) -> Dict[str, Any]:
        return {
            '_type': 'process',
            '_inputs': self.inputs(),
            '_outputs': self.outputs()}

    @composition.setter
    def composition(self, composition):
        pass

    @property
    def state(self) -> Dict[str, Any]:
        return self

    @state.setter
    def state(self, state):
        pass

    def initial_state(self):
        """Return initial state values, if applicable."""
        return {}

    def inputs(self):
        """
        Return a dictionary mapping input port names to bigraph types.

        Example:
            {'glucose': 'float', 'biomass': 'map[float]'}
        """
        return self.run_command('inputs', ())

    def outputs(self):
        """
        Return a dictionary mapping output port names to bigraph types.

        Example:
            {'growth_rate': 'float'}
        """
        return self.run_command('outputs', ())

    def invoke(self, state, interval):
        result = self.run_command('update', (state, interval))
        return SyncUpdate(result)

    def update(self, state, interval):
        return self.run_command('update', (state, interval))

    def end(self) -> None:
        """
        remove the container
        """
        # Only end once.
        if self._ended:
            return

        self.socket.close()

        self.container.stop()
        self.container.remove()

        self._ended = True

    def __del__(self) -> None:
        self.end()

def initialize_docker():
    client: DockerClient = None
    try:
        client = docker.from_env()
    except Exception:
        pass # We handle this by checking if client is None elsewhere.
    return client

class DockerProtocol(Protocol):
    client = initialize_docker()

    @classmethod
    def interface(cls, core, data):
        if cls.client is None:
            raise NotImplementedError("Docker was unable to be initialized; check your installation and try again.")
        image = cls.client.images.get(
            data['image'])

        config_schema = json.loads(
            image.labels['config_schema'])

        def instantiate(config, core=None):
            return DockerProcess(
                cls.client,
                data,
                config,
                core)

        instantiate.config_schema = config_schema
        return instantiate