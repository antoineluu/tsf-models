"""Get model's parameters."""

from json import loads


class ParametersGen:
    """
    Get parameters that are fixed.
    (not including trainable parameters).
    """

    def get_params(self, o):
        layers_metadata = [loads(layer._tracking_metadata)["config"] for layer in o.model.layers]
        dict = {"model_name": o.__class__.__name__, "layers": layers_metadata}
        return dict
