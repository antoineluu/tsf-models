"""Operations for reading and writing files into the database."""

from data_access.read_db import DataAccessRead
from data_access.write_db import DataAccessWrite
from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class DataAccessController:
    """
    Controller class for MongoDB.
    Use for saving model architectures and retrieving model list.

    Args:
    -------
    url: String
        URL to connect to the database. Normally, the URL is configured
        in the Config object.
    dar: DataAccessRead
        Responsible for reading data.
    daw: DataAccessWrite
        Responsible for writing data.
    client: MongoClient
        Client for the MongoDB instance.
    """

    def __init__(self, __C):
        # self.url = __C.MODEL_DB_URL
        self.url = "localhost:27017"
        self.dar = DataAccessRead()
        self.daw = DataAccessWrite()
        self.client = None

    def connect(self):
        """Establish a connection."""
        try:
            if not self.check_connection(raise_exception=False):
                self.client = MongoClient(self.url)

                # Create local "nobel" database on the fly
                self.db = self.client["db"]
                # Create "model" collection
                self.model_collection = self.db["model"]

                logger.info("[MODEL DAC] Connect successfully.")

        except ConnectionFailure:
            raise ConnectionError("[MODEL DAC] Cannot connect to the model database.")

    def check_connection(self, raise_exception=True):
        """Check if the application has establish a connection to the database."""
        if raise_exception and self.client is None:
            raise Exception("[MODEL DAC] You have not establish a connection.")
        return self.client is not None

    def is_model_exists(self, model_name):
        """Check whether a model is exists in the database by its name."""
        self.connect()
        if self.check_connection():
            return self.dar.is_model_exists(self.model_collection, model_name)

    def insert_model(self, model_arch, overwrite=False):
        """
        Insert a new model.

        Args:
        -------
        model_name: String
            Model name.
        model_arch: Json
            The architecture which is saved in JSON format.

        Returns:
        -------
        A boolean variable indicating the success of insert process.
        """
        self.connect()
        if self.check_connection():
            model_name = model_arch["model_name"]
            if self.is_model_exists(model_name):
                if overwrite:
                    self.daw.update_model(self.model_collection, model_name, model_arch)
                else:
                    logger.info(
                        "[MODEL DAC] Cannot insert the {} model since it is "
                        "already exists, to overwrite the model architecture, "
                        "set 'overwrite' argument to True.".format(model_name)
                    )
                    return False
            else:
                return self.daw.insert_model(self.model_collection, model_name, model_arch)
