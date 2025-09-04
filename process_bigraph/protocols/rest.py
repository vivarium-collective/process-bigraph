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

from urllib.parse import urlparse, urlunparse
import requests

from process_bigraph.composite import Process, SyncUpdate
from process_bigraph.protocols.protocol import Protocol
from process_bigraph.protocols.local import LocalProtocol


def rest_get(url, parameters=None):
    return requests.get(
        urlunparse(url),
        json=parameters).json()


def rest_post(url, parameters=None):
    return requests.post(
        urlunparse(url),
        json=parameters).json()


class RestProcess(Process):
    def __init__(self, data, config, core) -> None:
        self._ended = False
        self.base_url = data['base_url'] or urlparse('http://localhost:22222')
        self.initialize_url = self.base_url._replace(path='/initialize')
        self.inputs_url = self.base_url._replace(path='/inputs')
        self.outputs_url = self.base_url._replace(path='/outputs')
        self.update_url = self.base_url._replace(path='/update')

        self.process_name = data['process_name']

        self.process_id = rest_post(
            self.initialize_url,
            config)

        super().__init__(config, core=core)

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
        inputs_url = self.base_url._replace(
            path=f'/inputs/{self.process_id}')
        response = rest_get(inputs_url)

        return response

    def outputs(self):
        """
        Return a dictionary mapping output port names to bigraph types.

        Example:
            {'growth_rate': 'float'}
        """
        outputs_url = self.base_url._replace(
            path=f'/outputs/{self.process_id}')
        response = rest_get(outputs_url)

        return response

    def update(self, state, interval):
        update_url = self.base_url._replace(
            path=f'/update/{self.process_id}')
        response = rest_post(update_url, {
            'state': state,
            'interval': interval})
        return response

    def end(self) -> None:
        """
        remove the container
        """
        # Only end once.
        if self._ended:
            return

        end_url = self.base_url._replace(
            path=f'/end/{self.process_id}')
        rest_post(end_url)

        self._ended = True

    def __del__(self) -> None:
        self.end()


class RestProtocol(Protocol):
    @classmethod
    def interface(cls, core, data):
        ssh = ''
        process_name = data['process']
        host = data['host']
        port = data['port']
        base_raw = f'http{ssh}://{host}:{port}'
        base_url = urlparse(base_raw)

        config_schema_url = base_url._replace(path='/config-schema')
        config_schema = rest_get(
            config_schema_url)

        instant = {
            'base_url': base_url,
            'process_name': process_name}

        def instantiate(config, core=None):
            return RestProcess(
                instant,
                config,
                core)

        instantiate.config_schema = config_schema
        return instantiate
