"""Reading operations."""
from loguru import logger


class DataAccessRead:
    def is_model_exists(self, collection, model_name):
        """Check whether a model exists in the database."""

        # Inspect the collection
        logger.info("[MODEL DAR] The model collection has {} documents.".format(collection.count_documents({})))

        count = collection.count_documents({"model_name": model_name})
        return count != 0
