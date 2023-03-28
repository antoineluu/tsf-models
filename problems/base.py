from typing import Dict


class BaseProblem:
    """
    Define a Machine Learning problem (can extend to an Analysis problem later).
    There are currently 3 problems available, implementing the first one.
    1. Time Series problems (include Unvariate and Multivariate problems)
    2. Regression problems.
    3. Classification problems.
    """

    def __init__(self):
        """
        The constructor specifies the dataset and used columns in the dataset.
        The subclass's constructor can add other arguments.
        """
        pass

    def get_best_model(self):
        """Return the best model for the data."""
        pass

    def get_dict(self, dict) -> Dict:
        """Return the dictionary representing problem configuration (metadata)"""
