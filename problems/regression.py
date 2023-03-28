from typing import List, Dict

import numpy as np
import pandas as pd
from forecasting_module.dataset.reg_dataset import RegressionDataset
# from forecasting_module.models.linear import BasicLinearRegression
from forecasting_module.problems.base import BaseProblem


class RegressionProblem(BaseProblem):
    """
    A timeseries problem is a problem that uses attributes from the history
    to infer values in the future.
    """

    def __init__(
        self,
        dataset: RegressionDataset,
        label: str,
        date_col: str,
        pipeline_id: int = None,
        time_lag: str = '5s',
        input_cols: List[str] = None,
        first_config: bool = False,
        **kwargs
    ):
        """
        Define regression problem.

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
        super(RegressionProblem, self).__init__()
        self.date_col = date_col
        self.input_cols = set(input_cols)
        self.label = label
        self.pipeline_id = pipeline_id
        self.time_lag = time_lag

        dataset.set_cols(input_cols, label)

        if first_config:
            # Check if defined columns are exists
            dataset.check_columns_exists(label=self.label, date_col=self.date_col, input_cols=self.input_cols)

        # Currently accept numeric columns only
        # Future step: add categorical columns and perform dummy encoding 
        self.input_cols = dataset.data[input_cols].select_dtypes(include=np.number).columns.tolist()

        # Save the best model for this problem
        self.best_model = self.get_best_model()

    def get_best_model(self):
        return BasicLinearRegression

    def get_dict(self) -> Dict:
        meta = {
            "id": self.pipeline_id,
            "label": self.label,
            "date_col": self.date_col,
            "time_lag": self.time_lag,
            "type": "Regression"
        }
        for i in range(len(self.input_cols)):
            lst_input_cols = list(self.input_cols)
            meta[f"input:{i}"] = lst_input_cols[i]
        return meta
