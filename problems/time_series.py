from typing import Dict, List

from forecasting_module.dataset.ts_dataset import TimeSeriesDataset
from forecasting_module.models.base_model import BaseModel
from forecasting_module.models.dilatedcnn import DilatedCNN
from forecasting_module.problems.base import BaseProblem


class TimeSeriesProblem(BaseProblem):
    """
    A timeseries problem is a problem that uses attributes from the history
    to infer values in the future.
    """

    def __init__(
        self,
        label: str,
        date_col: str,
        input_cols: List[str],
        pipeline_id: int = None,
        sampling_rule: str = "5min",
        time_lag: str = "5s",
        input_timesteps: int = 14,
        output_timesteps: int = 7,
        first_config: bool = False,
        # **kwargs,
    ):
        """
        Define a time series problem.

        Args:
        -------
        data: DataFrame
            Input data
        label: int
            Label column represented in ID.
        date_col: String
            Date time column
        input_cols: List[int]
            Input column(s) represented in ID.
        date_format: optional, String. Default: None
            The strftime to parse time, eg “%d/%m/%Y”, note that “%f” will parse all the way up to nanoseconds.
            See strftime documentation for more information on choices:
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
            (WARNING: Unavailable in this stage, assume that all will follow ISO format %Y-%d-%m).
        sampling_rule: str. Default: '3min'
            The offset string or object representing target conversion.
        input_timesteps: int. Default: 24
            Number of historical timesteps as input.
        output_timesteps: int. Default: 24
            Number of future timesteps as predicted.
        first_config: bool. Default: False
            Set to True if this is the first configuration with the data.
        """
        super(TimeSeriesProblem, self).__init__()
        self.label = label
        self.input_cols = input_cols
        self.date_col = date_col
        self.sampling_rule = sampling_rule
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.time_lag = time_lag
        self.multi_var = False
        self.pipeline_id = pipeline_id

        if len(self.input_cols) > 1:
            self.multi_var = True

        # # Set column for later slicing (during get_train and get_val)
        # dataset.set_cols(input_cols + [self.label, self.date_col])

        # if first_config:
        #     # Check if defined columns are exists
        #     dataset.check_columns_exists(label=self.label, date_col=self.date_col, input_cols=self.input_cols)

        # Save the best model for this problem
        self.best_model = self.get_best_model()

    def get_best_model(self) -> BaseModel:
        return DilatedCNN

    def get_dict(self) -> Dict:
        meta = {
            "pipeline_id": self.pipeline_id,
            "label": self.label,
            "date_col": self.date_col,
            "input_cols": self.input_cols,
            "sampling_rule": self.sampling_rule,
            "time_lag": self.time_lag,
            "input_timesteps": self.input_timesteps,
            "output_timesteps": self.output_timesteps,
            "problem_type": "Time Series"
        }
        # for i in range(len(self.input_cols)):
        #     lst_input_cols = list(self.input_cols)
        #     meta[f"input:{i}"] = lst_input_cols[i]
        return meta