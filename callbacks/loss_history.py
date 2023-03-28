import asyncio
from types import coroutine

from api_module.services.training_management_service import TrainingManager, get_training_manager
from forecasting_module.callbacks.base import Callback


# nest_asyncio.apply()
class LossHistory(Callback):
    """Capturing the loss after each epoch during training."""

    def __init__(self, pipeline_id, **kwargs):
        self.pipeline_id = pipeline_id
        # self.training_manager = get_training_manager()

    def on_train_begin(self, logs, **kwargs):
        self.losses = []
        # message = {"status": "start", "value": "training started"}
        message = "start"

        # self.training_manager.push_message(self.pipeline_id, message)

    def on_epoch_end(self, epoch, logs,**kwargs):
        epoch_loss = logs.get("loss")
        epoch_loss = round(epoch_loss, 5)
        self.losses.append(epoch_loss)
        message = {"status": "training", "value": epoch_loss}
        # message = "epoch_loss "+str(epoch_loss)
        # self.training_manager.push_message(self.pipeline_id, message)

    def on_train_end(self, logs, **kwargs):
        # message = {"status": "done", "value": "training done"}
        logs['loss_history']=self.losses
        message = "done"
        # self.training_manager.push_message(self.pipeline_id, message)    