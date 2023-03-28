import asyncio
from types import coroutine
import numpy as np
from copy import deepcopy
import torch
from api_module.services.training_management_service import TrainingManager, get_training_manager
from forecasting_module.callbacks.base import Callback


# nest_asyncio.apply()
class ReturnBestModel(Callback):
    """Stops training when validation loss increases"""

    def __init__(self, pipeline_id, **kwargs):
        self.pipeline_id = pipeline_id
        self.training_manager = get_training_manager()


    def on_train_begin(self, logs, model,**kwargs):

        logs["training_model_dict"] = model.state_dict()

    def on_epoch_end(self, epoch, logs, **kwargs):
        val_loss = round(logs.get("val_loss"), 4)
        # message = "epoch_val_loss " +str(val_loss)
        # self.training_manager.push_message(self.pipeline_id, message)

        if val_loss <= logs.get("lowest_val_loss",val_loss):
            logs["lowest_val_loss"] = val_loss
            logs["best_model_dict"] = deepcopy(logs["training_model_dict"]), epoch

    def on_train_end(self, logs, model, **kwargs):
        model.load_state_dict(logs.get("best_model_dict")[0])
        print("returning best model at epoch", logs["best_model_dict"][1])