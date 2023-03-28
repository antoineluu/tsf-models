import asyncio
from types import coroutine
import numpy as np
from copy import deepcopy
import torch
from api_module.services.training_management_service import TrainingManager, get_training_manager
from forecasting_module.callbacks.base import Callback
import optuna

# nest_asyncio.apply()


class ReduceLROnPlateau(Callback):
    """Stops training when validation loss increases"""

    def __init__(self, pipeline_id, patience=3, ratio=0.5, minimum=1e-8, **kwargs):
        self.pipeline_id = pipeline_id
        self.training_manager = get_training_manager()
        self.patience = patience
        self.ratio = ratio
        self.minimum = minimum

    def on_train_begin(self, logs, **kwargs):
        pass

    def on_epoch_end(self, logs, optimizer, **kwargs):
        val_loss = round(logs.get("val_loss"), 4)

        if val_loss <= logs.get("lowest_val_loss", val_loss):
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience and optimizer.param_groups[0]['lr'] > self.minimum:
            optimizer.param_groups[0]['lr'] *= self.ratio
            print("updated lr", optimizer.param_groups[0]['lr'])
            self.counter = 0

    def on_train_end(self, logs, **kwargs):
        pass
