import asyncio
from types import coroutine
import numpy as np
from copy import deepcopy
import torch
from api_module.services.training_management_service import TrainingManager, get_training_manager
from forecasting_module.callbacks.base import Callback
import optuna

# nest_asyncio.apply()
class PruneTrial(Callback):
    """Stops training when validation loss increases"""

    def __init__(self, pipeline_id, trial=None,**kwargs):
        self.pipeline_id = pipeline_id
        self.training_manager = get_training_manager()
        self.trial = trial
    def on_train_begin(self, logs, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs,model,**kwargs):
        val_loss = round(logs.get("val_loss"), 4)
        self.trial.report(val_loss, epoch)
        if self.trial.should_prune():
            logs["Keep_training"] = False
        # self.training_manager.push_message(self.pipeline_id, str(self.counter))

    def on_train_end(self, logs, **kwargs):
        pass  