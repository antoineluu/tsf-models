"""Get model's hyperparameters."""

import inspect


class HyperparametersGen:
    """
    Get hyperparameter from a model.
    Normally, these hyperparameters is configurated in __init__, fit and compile methods.

    Args:
    -------
        non_hyper: List<str>
            List of obligatory arguments which are not considered as hyperparameters.
    """

    non_hyper = ["X", "y", "X_train", "y_train", "X_val", "y_val", "self"]

    def get_hyperparams(self, instance):

        hyper = inspect.signature(instance).parameters
        hyper_dict = list()
        for key in hyper.keys():
            dict = {"name": hyper[key].name, "type": hyper[key].annotation.__name__, "default": hyper[key].default}
            hyper_dict.append(dict)
        return hyper_dict
