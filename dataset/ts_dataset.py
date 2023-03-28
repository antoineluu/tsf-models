from typing import List, Set

import numpy as np
import pandas as pd
from forecasting_module.dataset.base_dataset import BaseDataset
from loguru import logger


class TimeSeriesDataset(BaseDataset):
    """Contain the data's logic when creating a time-series problem."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.size = len(data)

        # self.data.columns = self.data.columns.map(lambda x: int(x) if x.isnumeric() else x)

    # def set_cols(self, cols: List):
    #     self.cols = list(set(cols))
    def set_datetime_index(self, date_col):
        """Resample and set index. All categorical variable will also be removed."""
        if date_col not in self.data.columns:
            raise KeyError("{} does not appear in your data columns".format(date_col))


        self.data.loc[:,date_col] = pd.to_datetime(self.data.loc[:,date_col].squeeze())

        self.data = self.data.set_index(date_col).sort_index()
    
    def check_columns_exists(self, label: str, date_col: str, input_cols: Set) -> bool:
        """Check whether a list of columns is exists in the dataset or not."""
        input_cols.add(label)
        input_cols.add(date_col)
        value = pd.Index(input_cols)
        not_exist = value.difference(self.data.columns)
        if len(not_exist) > 0:
            logger.info("[FORECASTING MODULE] {} does not appear in your data columns".format(not_exist))
            raise KeyError("{} does not appear in your data columns, require {}".format(not_exist, self.data.columns))
        return True

    # def remove_null_rows(self, used_column):
    #     """DEPRECIATED: Integrated within preprocessor."""
    #     self.data = self.data[used_column].dropna(how="all")

    # def set_index(self, date_col) -> pd.DataFrame:
    #     """Resample and set index. All categorical variable will also be removed.
    #     DEPRECIATED: Integrated within preprocessor."""
    #     if date_col not in self.data.columns:
    #         logger.info("[FORECASTING MODULE] {} does not appear in your data columns".format(date_col))
    #         raise KeyError("{} does not appear in your data columns".format(date_col))

    #     try:
    #         # Check if the column can be converted to datetime
    #         self.data[date_col] = pd.to_datetime(self.data[date_col])
    #     except:
    #         logger.info(
    #             "[FORECASTING MODULE] The expected date time column does not have the correct format. \
    #             Try to choose different column or change its format."
    #         )
    #         raise ValueError(
    #             "The expected date time column does not have the correct format. \
    #             Try to choose different column or change its format."
    #         )
    #     self.data = self.data.set_index(date_col).sort_index()

    # def validate(self):
    #     missing_response = self.check_missing()
    #     # TODO: Add some more validation here
    #     return missing_response

    # def check_missing(self):
    #     missing_df = self.data.isna().mean(axis=0).reset_index()
    #     missing_df.columns = ["Category", "Value"]
    #     missing_df["Comment"] = missing_df["Value"].transform(self.comment_on_missing)
    #     missing_df["Category"] = missing_df["Category"].transform(lambda x: x + " (Missing Percentage)")
    #     return missing_df

    # def comment_on_missing(self, value):
    #     return "OK" if value < 0.5 else "There is too much missing data"

    def get_train_val_idx(self, val_size: float = 0.2, test_size: float = 0.2):
        """Get indices for training and validation set."""
        n_1 = int(len(self.data) * (1 - val_size - test_size))
        n_2 = int(len(self.data) * (1 - test_size))
        train_idx = np.arange(n_1)
        val_idx = np.arange(n_1, n_2)
        test_idx = np.arange(n_2, len(self.data))

        return train_idx, val_idx, test_idx

    def get_data(self):
        return self.data.copy()

    def get_train(self):
        if not hasattr(self, "train_idx"):
            self.train_idx, self.val_idx, self.test_idx = self.get_train_val_idx()
        return self.data.iloc[self.train_idx].copy()

    def get_val(self):
        if not hasattr(self, "val_idx"):
            self.train_idx, self.val_idx, self.test_idx = self.get_train_val_idx()
        return self.data.iloc[self.val_idx].copy()

    def get_test(self):
        if not hasattr(self, "val_idx"):
            self.train_idx, self.val_idx, self.test_idx = self.get_train_val_idx()
        return self.data.iloc[self.test_idx].copy()

    # def align_time(self, sampling_rule: str, time_lag: str, limit: int = None):
    #     """Align time to sampling_rule.
    #     DEPRECIATED: Integrated within preprocessor."""
    #     base = "1H" if sampling_rule.endswith("H") else sampling_rule
    #     index = pd.date_range(
    #         start=self.data.index[0].round(base), end=self.data.index[-1].round(base), freq=sampling_rule
    #     )
    #     self.data = self.data.apply(lambda x: x.dropna().reindex(index, method="nearest", tolerance=time_lag))
    #     if limit is not None:
    #         self.data = self.data[-limit:]
