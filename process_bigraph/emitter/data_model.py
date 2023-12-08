import copy
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pymongo.errors import DocumentTooLarge
from pymongo.mongo_client import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bigraph_schema import get_path, set_path
from process_bigraph.core.composite import Step
from process_bigraph.emitter.utils import (
    HISTORY_INDEXES,
    CONFIGURATION_INDEXES,
    make_fallback_serializer_function,
    serialize_value,
    breakdown_data,
    assoc_path,
    get_history_data_db
)


class Emitter(Step):
    """Base emitter class. An `Emitter` implementation instance diverts all querying of data to
        the primary historical collection whose type pertains to Emitter child, i.e:
            database-emitter=>`pymongo.Collection`, ram-emitter=>`.RamEmitter.history`(`List`)
    """
    def query(self, query=None):
        return {}

    def update(self, state) -> Dict:
        return {}


class ConsoleEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def schema(self) -> Dict:
        return self.config['ports']


    def update(self, state) -> Dict:
        print(state)
        return {}


class RAMEmitter(Emitter):
    config_schema = {
        'ports': 'tree[any]'}


    def __init__(self, config):
        super().__init__(config)

        self.history = []


    def schema(self) -> Dict:
        return self.config['ports']


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


class DatabaseEmitter(Emitter):
    """
        Emit data to a mongoDB database

        Attributes:
            client_dict:`Dict[str, MongoClient]`

        >>> # The line below works only if you have to have 27017 open locally
        >>> # emitter = DatabaseEmitter(config)

        PLEASE NOTE: For Mac Silicon, you can start a Mongo instance as a background process with:

            ``mongod --config /opt/homebrew/etc/mongod.conf --fork``

            ...and the process can be stopped in two ways:

                1.) `kill {PID}`

                2.) `mongosh -> use admin -> db.shutdownServer()`
    """
    client_dict: Dict[int, MongoClient] = {}
    config_schema = {
        'ports': 'tree[any]',
        'experiment_id': {
            '_type': 'string',
            '_default': str(uuid.uuid4())
        },
        'emit_limit': {
            '_type': 'int',
            '_default': 4000000
        },
        'embed_path': {
            '_type': 'tuple',
            '_default': tuple(),
            '_deserialize': 'deserialize_string'
        },
        'host': {
            '_type': 'string',
            '_default': 'localhost:27017'
        },
        'database': {
            '_type': 'string',
            '_default': 'simulations'
        },
    }


    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Config may have 'host' and 'database' items. The config passed is expected to be:

                {'experiment_id':,
                 'emit_limit':,
                 'embed_path':,
                 'ports': {}}

                TODO: Automate this process for the user in builder
        """
        super().__init__(config)
        self.experiment_id = self.config['experiment_id']
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = self.config['emit_limit']
        self.embed_path = self.config['embed_path']

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in DatabaseEmitter.client_dict:
            DatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config['host'])
        self.client: MongoClient = DatabaseEmitter.client_dict[curr_pid]

        # extract objects from current mongo client instance
        self.db: Database = getattr(self.client, self.config.get('database', 'simulations'))
        self.history: Collection = getattr(self.db, 'history')
        self.configuration: Collection = getattr(self.db, 'configuration')
        self.phylogeny: Collection = getattr(self.db, 'phylogeny')

        # create column indexes for the given collection objects
        self.create_indexes(self.history, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)
        self.create_indexes(self.phylogeny, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function()

    def write_emit(self, table: Collection, emit_data: Dict[str, Any]) -> None:
        """Check that data size is less than emit limit. Break up large emits into smaller pieces and
            emit them individually.

            Args:
                table:`pymongo.collection.Collection`: pymongo collection to which data will be written.
                emit_data:`Dict[str, Any]`: Data to be passed and saved to the collection.
        """
        assembly_id = str(uuid.uuid4())
        emit_data = serialize_value(emit_data, self.fallback_serializer)
        try:
            emit_data['assembly_id'] = assembly_id
            table.insert_one(emit_data)
        # If document is too large, break up into smaller dictionaries
        # with shared assembly IDs and time keys
        except DocumentTooLarge:
            emit_data.pop('assembly_id')
            # insert_one adds this key to emit_data
            emit_data.pop('_id')
            experiment_id = emit_data.pop('experiment_id')
            time = emit_data['data'].pop('time', None)
            broken_down_data = breakdown_data(self.emit_limit, emit_data)
            for (path, datum) in broken_down_data:
                d: Dict[str, Any] = {}
                assoc_path(d, path, datum)
                d['assembly_id'] = assembly_id
                d['experiment_id'] = experiment_id
                if time:
                    d.setdefault('data', {})
                    d['data']['time'] = time
                table.insert_one(d)

    def query(self, query: Optional[List[Tuple[str]]] = None) -> Dict:
        """API contract-wrapper for `self.get_data`. Get data based on the passed query.

             Args:
                 query: a list of tuples pointing to fields within the experiment data.
                     In the format: [('path', 'to', 'field1'), ('path', 'to', 'field2')]

            Returns:
                 `Dict`
        """
        return get_history_data_db(self.history, self.experiment_id, query)

    def update(self, state):
        if not state.get('table'):
            state['table'] += self.config['database']
        table_id = state['table']
        table = self.db.get_collection(table_id)
        time = state['data'].pop('time', None)
        state['data'] = assoc_path({}, self.embed_path, state['data'])
        # Analysis scripts expect the time to be at the top level of the
        # dictionary, but some emits, like configuration emits, lack a
        # time key.
        if time is not None:
            state['data']['time'] = time
        emit_data = state.copy()
        emit_data.pop('table', None)
        emit_data['experiment_id'] = self.experiment_id
        self.write_emit(table, emit_data)
        print(emit_data)
        return {}

    def schema(self) -> Dict:
        return self.config['ports']
