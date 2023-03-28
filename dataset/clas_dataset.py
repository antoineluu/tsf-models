import numpy as np
import pandas as pd
from forecasting_module.dataset.base_dataset import BaseDataset
from loguru import logger


class ClassificationDataset(BaseDataset):
    """Contain the data's logic when creating a classification problem."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def set_cols(self, input_cols, target_col):
        self.input_cols = input_cols
        self.target_col = target_col

    def set_index(self, date_col):
        """Resample and set index."""

        if date_col not in self.data.columns:
            logger.info("[FORECASTING MODULE] {} does not appear in your data columns".format(date_col))
            raise KeyError("{} does not appear in your data columns".format(date_col))

        self.data[date_col] = pd.to_datetime(self.data[date_col])

        try:
            # Check if the column can be converted to datetime
            self.data[date_col] = pd.to_datetime(self.data[date_col])
        except:
            logger.info(
                "[FORECASTING MODULE] The expected date time column does not have the correct format. \
                Try to choose different column or change its format."
            )
            raise ValueError(
                "The expected date time column does not have the correct format. \
                Try to choose different column or change its format."
            )

        self.data = self.data.set_index(date_col).sort_index()

    def get_train_val_idx(self, val_size=0.2):
        arr_rand = np.random.rand(self.data.shape[0])
        split = arr_rand < np.percentile(arr_rand, (1 - val_size) * 100)
        return split

    def get_train(self) -> pd.DataFrame:
        if not hasattr(self, "idx"):
            self.idx = self.get_train_val_idx()
        take = self.data[self.idx]
        return take[self.input_cols], take[self.target_col]

    def get_val(self) -> pd.DataFrame:
        if not hasattr(self, "idx"):
            self.idx = self.get_train_val_idx()
        take = self.data[~self.idx]
        return take[self.target_col], take[self.target_col]
