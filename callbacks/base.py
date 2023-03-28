class Callback:
    def on_train_begin(self, logs=None, **kwargs):
        pass

    def on_train_end(self, logs=None, **kwargs):
        pass

    def on_batch_begin(self, batch, logs=None, **kwargs):
        pass

    def on_batch_end(self, batch, logs=None, **kwargs):
        pass

    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs=None, **kwargs):
        pass

    def on_predict_batch_begin(self, batch, logs=None, **kwargs):
        pass

    def on_predict_batch_end(self, batch, logs=None, **kwargs):
        pass
