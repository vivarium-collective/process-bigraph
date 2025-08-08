"""
===============================================
Protocol for running processes in parallel using
python multiprocessing
===============================================
"""

import sys
import pstats
from typing import Any, Dict, Optional, Union, List, Tuple

import multiprocessing
from multiprocessing.connection import Connection

from process_bigraph.composite import Process
from process_bigraph.protocols.protocol import Protocol
from process_bigraph.protocols.local import LocalProtocol

def _handle_parallel_process(
        connection: Connection, process: Process,
        profile: bool) -> None:
    '''Handle a parallel Vivarium :term:`process`.

    This function is designed to be passed as ``target`` to
    ``Multiprocess()``. In a loop, it receives :term:`process commands`
    from a pipe, passes those commands to the parallel process, and
    passes the result back along the pipe.

    The special command ``end`` is handled directly by this function.
    This command causes the function to exit and therefore shut down the
    OS process created by multiprocessing.

    Args:
        connection: The child end of a multiprocessing pipe. All
            communications received from the pipe should be a 3-tuple of
            the form ``(command, args, kwargs)``, and the tuple contents
            will be passed to :py:meth:`Process.run_command`. The
            result, which may be of any type, will be sent back through
            the pipe.
        process: The process running in parallel.
        profile: Whether to profile the process.
    '''
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
    running = True

    while running:
        command, args, kwargs = connection.recv()

        if command == 'end':
            running = False
        else:
            result = process.run_command(command, args, kwargs)
            connection.send(result)

    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        connection.send(stats.stats)  # type: ignore

    connection.close()


class ParallelProcess(Process):
    def __init__(
            self, process: Process, profile: bool = False,
            stats_objs: Optional[List[pstats.Stats]] = None) -> None:
        """Wraps a :py:class:`Process` for multiprocessing.

        To run a simulation distributed across multiple processors, we
        use Python's multiprocessing tools. This object runs in the main
        process and manages communication between the main (parent)
        process and the child process with the :py:class:`Process` that
        this object manages.

        Most methods pass their name and arguments to
        :py:class:`Process.run_command`.

        Args:
            process: The Process to manage.
            profile: Whether to use cProfile to profile the subprocess.
            stats_objs: List to add cProfile stats objs to when process
                is deleted. Only used if ``profile`` is true.
        """

        self._ended = False
        self._pending_command: Optional[
            Tuple[str, Optional[tuple], Optional[dict]]] = None
        self.process = process

        super().__init__(process.config, core=process.core)

        self.profile = profile
        self._stats_objs = stats_objs
        assert not self.profile or self._stats_objs is not None
        # Linux's default ``fork`` start method causes a lot of random
        # issues, including python/cpython#110770 (prompted this change)
        # and python/cpython#84559 (general discussion). This default
        # will be changed to ``forkserver`` in Python 3.14. MacOS and
        # Windows use the much safer but slightly slower ``spawn`` method
        if sys.platform not in ("darwin", "win32"):
            start_method = "forkserver"
        else:
            start_method = "spawn"
        mp_ctx = multiprocessing.get_context(start_method)
        self.parent, child = mp_ctx.Pipe()
        self.multiprocess = mp_ctx.Process( # type: ignore[attr-defined]
            target=_handle_parallel_process,
            args=(child, process, self.profile))
        self.multiprocess.start()

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
        self.parent.send((command, args, kwargs))

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
        return self.parent.recv()

    def get(self):
        return self.get_command_result()

    @staticmethod
    def generate_state(config=None):
        """Generate static initial state for user configuration or inspection."""
        return type(self.process).generate_state(config)

    @property
    def config(self) -> Dict[str, Any]:
        return self.run_command('config')

    @config.setter
    def config(self, config):
        self.run_command('set_config', (config,))

    @property
    def composition(self) -> Dict[str, Any]:
        return self.run_command('composition')

    @composition.setter
    def composition(self, composition):
        self.run_command('set_composition', (composition,))

    @property
    def state(self) -> Dict[str, Any]:
        return self.run_command('state')

    @state.setter
    def state(self, state):
        self.run_command('set_state', (state,))

    def initial_state(self):
        """Return initial state values, if applicable."""
        return self.run_command('initial_state', ())

    def inputs(self):
        """
        Return a dictionary mapping input port names to bigraph types.

        Example:
            {'glucose': 'float', 'biomass': 'map[float]'}
        """
        # return self.run_command('inputs', ())
        return self.process.inputs()

    def outputs(self):
        """
        Return a dictionary mapping output port names to bigraph types.

        Example:
            {'growth_rate': 'float'}
        """
        return self.process.outputs()
        # return self.run_command('outputs', ())

    def invoke(self, state, interval):
        self.send_command('update', (state, interval))
        return self

    def update(self, state, interval):
        return self.run_command('update', (state, interval))

    def end(self) -> None:
        """End the child process.

        If profiling was enabled, then when the child process ends, it
        will compile its profiling stats and send those to the parent.
        The parent then saves those stats in ``self.stats``.
        """
        # Only end once.
        if self._ended:
            return
        self.send_command('end')
        if self.profile:
            stats = pstats.Stats()
            stats.stats = self.get_command_result()  # type: ignore
            assert self._stats_objs is not None
            self._stats_objs.append(stats)
        self.multiprocess.join()
        self.multiprocess.close()
        self._ended = True

    def __del__(self) -> None:
        self.end()


class ParallelProtocol(Protocol):
    @staticmethod
    def interface(core, address):
        local_instantiate = LocalProtocol.interface(core, address)
        def instantiate(config, core=None):
            instance = local_instantiate(config, core=core)
            return ParallelProcess(instance)

        instantiate.config_schema = local_instantiate.config_schema
        return instantiate
