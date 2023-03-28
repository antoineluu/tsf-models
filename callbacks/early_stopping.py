import asyncio
from types import coroutine
import numpy as np
from copy import deepcopy
import torch
from api_module.services.training_management_service import TrainingManager, get_training_manager
from forecasting_module.callbacks.base import Callback


# nest_asyncio.apply()
class EarlyStopping(Callback):
    """Stops training when validation loss increases"""

    def __init__(self, pipeline_id, patience : int=10, **kwargs):
        self.pipeline_id = pipeline_id
        self.patience = patience
        # self.training_manager = get_training_manager()
    def on_train_begin(self, logs, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs,**kwargs):
        val_loss = round(logs.get("val_loss"), 4)
        if val_loss <= logs.get("lowest_val_loss",val_loss):
            self.counter = 0
        else :
            self.counter += 1
            # self.training_manager.push_message(self.pipeline_id, str(self.counter))
        if self.counter >= self.patience:
            logs["Keep_training"] = False

    def on_train_end(self, logs, **kwargs):
        pass  