from typing import List

import numpy as np
import pandas as pd
from forecasting_module.dataset.reg_dataset import RegressionDataset
# from forecasting_module.models.random_forest_classifier import RandomForestClassification
from forecasting_module.problems.base import BaseProblem


class ClassificationProblem(BaseProblem):
    """
    A timeseries problem is a problem that uses attributes from the history
    to infer values in the future.
    """

    def __init__(
        self,
        dataset: RegressionDataset,
        label: str,
        date_col: str,
        input_cols: List[str] = None,
        first_config: bool = False,
        **kwargs
    ):
        """
        Define input data.

        Args:
        -------
        data: DataFrame
            Input data
        label: String
            Label column (Target column)
        input_cols: List of Strings
            List of input columns
        first_config: bool. Default: False
            Set to True if this is the first configuration with the data
        """
        super(ClassificationProblem, self).__init__()
        self.dataset = dataset
        self.label = pd.Index([label])

        # TODO: Accept object column also, then use one hot encoder for those.
        if input_cols is not None:
            self.input_cols = pd.Index(input_cols)
        else:
            self.input_cols = (
                dataset.data.select_dtypes(include=np.number).columns.drop(self.label)
                if input_cols is None
                else input_cols
            )

        if not first_config:
            dataset.set_index(date_col)

        # Save the best model for this problem
        self.best_model = self.get_best_model(dataset)

    def get_best_model(self, dataset):
        return RandomForestClassification
