import skl2onnx
import torch
from skl2onnx.common.data_types import FloatTensorType


class ModelFrameworkService:
    def save_model(self, filename, model, name):
        """
        Saving model to .ONNX format

        Args:
        -------
        filename: String
            Destination of the file.
        model: Model
            The model instance to be stored.
        name: String
            The model name.
        """
        raise NotImplementedError

    def save_model(self, output_path, model, **kwargs):
        raise NotImplementedError


class SklearnService(ModelFrameworkService):
    def save_model(self, output_path, model, name, **kwargs):
        # From http://onnx.ai/sklearn-onnx/index.html
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        onnx_model = skl2onnx.convert_sklearn(model, name, initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class PytorchService(ModelFrameworkService):
    def save_model(self, output_path, model, args, **kwargs):
        torch.onnx.export(model, args=args, f=output_path, **kwargs)
