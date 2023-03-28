from typing import Dict
import numpy as np
from forecasting_module.models.framework_utils.torch.dataloader import acquire_test_tensor
from forecasting_module.models.framework_utils.torch.dataloader import acquire_dataloader
from forecasting_module.problems.base import BaseProblem
import optuna
from optuna.visualization.matplotlib import plot_param_importances
from forecasting_module.models.framework_utils.torch.trainer import TorchTrainer
from torchsummary import summary
from optuna.visualization.matplotlib import plot_contour
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import re
import onnx


class BaseModel:
    """Abstract class for model implementation."""

    def __init__(self) -> None:
        """BaseModel constructor"""
        pass

    def _sample_params(self, p_name,  trial, layer=None):
        p_dict = self.upt_dict[p_name]
        for param_type in ["model_params", "training_params", "dataloader_params"]:
            self_p_dict = vars(self)[param_type]
            self_p_dict_tuning = vars(self)[param_type+"_tuning"]
            if p_name in self_p_dict:
                optuna_p_name = p_name
                if isinstance(layer, int):
                    optuna_p_name = p_name+"_"+str(layer)
                    p_dict = deepcopy(p_dict)
                    p_dict.pop("n_layers")
                elif layer == "n_layers":
                    optuna_p_name = p_name+"_n_layers"
                    self_p_dict_tuning = self_p_dict_tuning[p_name]
                    if p_dict["n_layers"] is None or (
                        p_dict["n_layers"]["type"] == "default_tuning" and 
                        "n_layers" not in self_p_dict_tuning):
                        p_dict = {"type": "constant",
                                  "params": len(self_p_dict[p_name])}
                    else:p_dict = p_dict["n_layers"]
                    p_name = "n_layers"
                if p_dict is None:
                    value = None
                elif "n_layers" in p_dict:
                    value = [self._sample_params(p_name, trial=trial, layer=i) for i in range(
                        self._sample_params(p_name, trial=trial, layer="n_layers"))]
                elif p_dict['type'] == "default_tuning":
                    value = self._suggest(
                        optuna_p_name, self_p_dict_tuning[p_name], trial) if p_name in self_p_dict_tuning else None
                elif p_dict['type'] == "constant":
                    value = p_dict["params"]
                elif p_dict['type'] == "bool":
                    value = trial.suggest_categorical(
                        optuna_p_name, [True, False])
                elif p_dict['type'] in ["float", "cate", "int"]:
                    value = self._suggest(optuna_p_name, p_dict, trial)
                if layer is not None:
                    return value
                if value is not None:
                    self_p_dict[p_name] = value

    def _suggest(self, p_name, dictio, trial):
        suggest_dict = {
            "float": trial.suggest_float,
            "cate": trial.suggest_categorical,
            "int": trial.suggest_int
        }
        if dictio["type"] == "cate":
            return suggest_dict[dictio["type"]](p_name, dictio["params"])
        else:
            return suggest_dict[dictio["type"]](p_name, dictio["params"]["lower"], dictio["params"]["upper"], **dictio.get("kparams", {}))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pipeline_id: str,
        verbose: bool = False,
    ):
        self.args = torch.randn(
            1, X_train.shape[1], X_train.shape[-1])
        input_params = dict(
            input_horizon=X_train.shape[-1],
            output_horizon=y_train.shape[-1],
            N_cols=X_train.shape[1])
        self.model_params.update(input_params)
        self.training_params["callbacks"]["LossHistory"] = {}
        self.training_params["callbacks"]["ValLossHistory"] = {}

        # self.model = self.model_class(**self.model_params)
        self.model = self.get_model(**self.model_params)
        self.trainer = TorchTrainer(
            **self.training_params, pipeline_id=pipeline_id, model=self.model, verbose=verbose)
        self.training_params["callbacks"].pop("LossHistory")
        self.training_params["callbacks"].pop("ValLossHistory")
        self.model_params.pop("input_horizon")
        self.model_params.pop("output_horizon")
        self.model_params.pop("N_cols")
        train_dataloader = acquire_dataloader(
            X_train, y_train, **self.dataloader_params)
        val_dataloader = () if X_val is None else acquire_dataloader(
            X_val, y_val, **self.dataloader_params)
        if verbose:
            for param in self.get_args(collapsed=True):
                print(param, self.get_args(collapsed=True)[param])
        res = self.trainer.train(train_dataloader, val_dataloader)

        return res
    
    def get_model(self, **kwargs):
        return self.model_class(**kwargs)
    
    def predict(self, X: np.ndarray, y):
        dataloader = acquire_dataloader(
            X,
            y,
            batch_size=self.dataloader_params['batch_size'],
            shuffle=False
        )
        return self.trainer.test(dataloader)

    def forward(self, X: np.ndarray):
        return self.model.forward(X)

    def optimize(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            pipeline_id: str,
            params,
            **kwargs):
        """ Instantiates a study for the present model then returns the best configuration.
        Trains over a fraction of the training data. A search space is given for each parameter as input."""
        n_trials = kwargs.get("n_trials", 5)
        timeout = kwargs.get("timeout", 7200)
        tuning_fraction = kwargs.get("tuning_fraction", 0.25)
        prune = kwargs.get("prune", True)
        index = np.int(len(X_train) - len(X_train)*tuning_fraction)

        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        self.upt_dict = params

        study.optimize(lambda trial: self.objective(trial, X_train[index:], y_train[index:],
                       X_val, y_val, pipeline_id, prune), n_trials=n_trials, timeout=timeout)

        trial = study.best_trial

        # add list parameters to model parameter dicts and delete layer parameters
        layers = [str(i) for i in range(trial.params.get("n_layers", 0))]
        for key, value in deepcopy(trial.params).items():
            if key[-1] in layers:
                layers[int(key[-1])] = value
                layer_key = key
                trial.params[layer_key[:-2]] = layers
                trial.params.pop(key)

        # replace layer parameters into list in the trial.param
        items = deepcopy(trial.params).items()
        pattern = "n_layers$"
        for key, value in items:
            if re.search(pattern, key):
                trial.params[key[:-9]] = [trial.params.pop(key[:-8]+str(i), self.get_args(
                    True)[key[:-9]][0]) for i in range(trial.params.pop(key))]
        # update parameters dicts
        self.model_params.update(
            {k: trial.params[k] for k in self.model_params if k in trial.params})
        self.training_params.update(
            {k: trial.params[k] for k in self.training_params if k in trial.params})
        self.dataloader_params.update(
            {k: trial.params[k] for k in self.dataloader_params if k in trial.params})
        return self.get_args()
        # plot_param_importances(
        #     study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        # )
        # plot_param_importances(study)
        # plot_contour(study, params=["learning_rate", "dropout_ratio"])
        # plt.show()

    def objective(
        self,
        trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        pipeline_id: str,
        prune: bool,
    ):
        """ Objective function that takes a trial object as input and returns the validation score.
        Parameters are sampled from the search space by the sampling method specified in the study."""

        # loop on parameter update list for optional sampling
        for p_name in self.upt_dict.keys():
            self._sample_params(p_name, trial=trial)

        # adding pruning callback and turn off return_best_model callback if there is one
        if prune:
            self.training_params["callbacks"]["PruneTrial"] = {"trial": trial}
        return_best_model = self.training_params["callbacks"].pop(
            "ReturnBestModel", None)
        logs = self.fit(X_train, y_train, X_val, y_val, pipeline_id)
        if prune:
            self.training_params["callbacks"].pop("PruneTrial")
        if not return_best_model is None:
            self.training_params["callbacks"]["ReturnBestModel"] = return_best_model
        if prune and trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return logs["lowest_val_loss"]

    def get_args(self, collapsed=None) -> any:
        """
        Return all training hyperparameters option.

        Returns:
        -------
        training_dict: Dict[str, str]
            A dictionary in format (hyperparameter_name, type/values)
        """
        if collapsed is not None:
            return {**self.training_params,**self.model_params,**self.dataloader_params}
        return {
            "training_params": self.training_params,
            "model_params": self.model_params,
            "dataloader_params": self.dataloader_params
        }

    def save_model(self, output_path):
        torch.onnx.export(
            self.model,
            opset_version=11,
            args=self.args if hasattr(self, "args") else None,
            f=output_path)
