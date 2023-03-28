class BaseDataset:
    def __init__(self):
        pass

    def validate_data(self):
        """Validate data based on the problem."""
        pass

    def set_index(self):
        """Set the datetime column as index."""
        pass

    def get_train_val_idx(sel):
        """Get indices to separate training and validation set."""
        pass

    def get_train(self):
        """Return training data."""
        pass

    def get_val(self):
        """Return validation data."""
        pass
