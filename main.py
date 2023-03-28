import io
import logging
import logging.handlers
import os
import pickle
import queue
import struct
import pandas as pd
import onnxruntime
import matplotlib.pyplot as plt
from api_module.model.config_request import ConfigTrainingRequest
from api_module.model.data_model import QueryDataRequest
from api_module.model.training_option import TrainRequest
from dataclasses import asdict

import numpy as np
from api_module.model.model_config import *
from api_module.model.problem_types import ProblemType
from forecasting_module.dataset.base_dataset import BaseDataset
from forecasting_module.factories.base_factory import BaseFactory
from forecasting_module.factories.factory import Factory
from forecasting_module.models import *
from forecasting_module.models.dilatedcnn import DilatedCNN
from forecasting_module.models.lstm import lstm
from forecasting_module.models.RLSTM import rlstm
from forecasting_module.models.gru import gru
from forecasting_module.models.tcn import tcn
from forecasting_module.models.nhits import NHITS
from forecasting_module.models.linear import Linear
from forecasting_module.models.nlinear import NLinear

from forecasting_module.params_gen.hyper_gen import HyperparametersGen
from forecasting_module.pipelines.base_pipeline import BasePipeline
from forecasting_module.problems.base import BaseProblem
from pandas import DataFrame

from fastapi import HTTPException
from copy import deepcopy


class ForecastingModule(object):
    """
    A service for ML functions.
    """
    factory: BaseFactory
    hyper_gen = HyperparametersGen()
    abstract_factory = Factory()

    def _model_mapping(self, request):
        mapping_dict = {
            DilatedCNNConfig: DilatedCNN,
            TCNConfig: tcn,
            NHITSConfig: NHITS,
            LSTMConfig: lstm,
            RLSTMConfig:rlstm,
            LinearConfig: Linear,
            NLinearConfig: NLinear,
            GRUConfig: gru,
            DilatedCNNTuning: DilatedCNN,
            TCNTuning: tcn,
            NHITSTuning: NHITS,
            LSTMTuning: lstm,
            RLSTMTuning: rlstm,
            LinearTuning: Linear,
            NLinearTuning: NLinear,
            GRUTuning: gru,
            
        }
        for config_type, model_type in mapping_dict.items():
            if isinstance(request, config_type):
                return model_type
        raise ModuleNotFoundError(
            "[FORECASTING MODULE] Cannot map request to model.")

    def _create_from_request(self, pipeline_config: dict, data: pd.DataFrame):
        """Define pipeline from training request"""
        self.factory = self.abstract_factory.get_factory(
            ProblemType(pipeline_config["config_problem"]["problem_type"]))
        dataset: BaseDataset = self.factory.create_dataset(data=data)
        problem: BaseProblem = self.factory.create_problem(
            label=pipeline_config["config_problem"]["label"],
            input_cols=pipeline_config["config_problem"]["input_cols"],
            date_col=pipeline_config["config_problem"]["date_col"],
            pipeline_id=pipeline_config["id"],
            input_timesteps=pipeline_config["config_problem"]["input_timesteps"],
            output_timesteps=pipeline_config["config_problem"]["output_timesteps"],
            time_lag=pipeline_config["config_problem"]["time_lag"],
            sampling_rule=pipeline_config["config_problem"]["sampling_rule"],
        )
        # dataset.set_datetime_index(problem.date_col)

        return problem, dataset

    def train(self, pipeline_config, data, request) -> None:

        problem, dataset = self._create_from_request(
            pipeline_config, data)
        model_class = self._model_mapping(request)

        # training request to params
        params = request.dict()
        if params["model_params"] == None:
            if "tuning_info" not in pipeline_config:
                params["model_params"] = {}
            elif params["model_name"] != pipeline_config["tuning_info"]["model_name"]:
                params["model_params"] = {}
            else:
                params["model_params"] = deepcopy(
                    pipeline_config["tuning_info"]["model_params"])
    
        param_sampling = {key: value for key,
                          value in {**params["model_params"] ,**params["training_params"]}.items() if value is not None}
        preprocessor_params = {key: value for key,
                          value in params["preprocessor_params"].items() if value is not None}
        model = model_class(**param_sampling)
        pipeline: BasePipeline = self.factory.create_pipeline(
            problem=problem, model=model, pipeline_id=pipeline_config["id"])
        pipeline.make_pipeline(transformers_params=preprocessor_params)
        logs = pipeline.fit(dataset.get_train(), dataset.get_val())

        result = pipeline.validate(dataset.get_test())
        result["parameter"] = np.array([problem.label] * len(result))
        return {"pipeline": pipeline, "result": result, "logs": logs}
        # return params, result, problem.get_dict(), model, pipeline.transformers, res

    def tune(self, pipeline_config, data, request) -> None:

        problem, dataset = self._create_from_request(pipeline_config, data)
        model = self._model_mapping(request)()

        pipeline: BasePipeline = self.factory.create_pipeline(
            problem=problem, model=model, pipeline_id=pipeline_config["id"])
        params = request.dict()
        tuning_params = {key: value for key,
                          value in params["tuning_params"].items() if value is not None}
        param_sampling = params["model_params_tuning"] | params["training_params_tuning"]
        preprocessor_params = params["preprocessor_params_tuning"]
        pipeline.make_pipeline(transformers_params=preprocessor_params)
        model_params = pipeline.tune_model(dataset.get_train(
        ), dataset.get_val(), params_update=param_sampling, **tuning_params)
        return {**{"tuning_params": tuning_params, "model_name": params["model_name"], "preprocessor_params": preprocessor_params}, **model_params}

    def forecast(self, pipeline_config, data, onnx_model, transformers):
        problem, dataset = self._create_from_request(pipeline_config, data)
        ort_session = onnxruntime.InferenceSession(onnx_model.getvalue())
        label = problem.label
        raw_data = dataset.get_data()
        preprocessed_data, cols = transformers.transform(
            raw_data, predict=True)
        raw_data = transformers.pipeline[0].transform(raw_data)
        rescaled_preprocessed = transformers.inverse_transform(
            preprocessed_data)[0][np.argwhere(cols == problem.label)]
        # rescaled_preprocessed = preprocessed_data[0][np.argwhere(cols == problem.label)]
        outputs = ort_session.run(
            None, {ort_session.get_inputs()[0].name: preprocessed_data})
        rescaled_forecast = transformers.inverse_transform(outputs[0])
        # rescaled_forecast = outputs[0]

        preprocessed_index = transformers.pipeline[-1].predict_index
        data = raw_data.loc[raw_data.index>=preprocessed_index.min()].copy()
        input_index = data.index
        indices = pd.date_range(
            start=data.index[-1],
            freq=problem.sampling_rule,
            periods=len(rescaled_forecast[0][0])+1)[1:]

        df_rescaled = pd.DataFrame(
            {"forecast": rescaled_forecast[0][0]}, indices)
        df_preprocessed = pd.DataFrame(
            {"preprocessed_rescaled_input": rescaled_preprocessed[0][0]}, preprocessed_index)
        df_input = pd.DataFrame(
            {"input": data.loc[:, problem.label].reindex(input_index)}
        )
        # print(df_rescaled.describe())
        # print(df_input.describe())

        df_input = data.loc[:, problem.label].to_frame().rename(columns={label: "input"})
        # print(df_input.describe(), df_input.shape[0], "nan", df_input.isna().sum(), "non nan", df_input.count())
        # print(df_preprocessed.describe(), df_preprocessed.shape[0], "nan", df_preprocessed.isna().sum(), "non nan", df_preprocessed.count())
        # print(df_preprocessed)
        return [df_input, df_preprocessed, df_rescaled]

    def config_problem(self, request: ConfigTrainingRequest, data) -> Dict:
        """
        Configure the problem, return list of parameters of the model that suits the data best.

        Args:
        -------
        request: Request
            Request for configure the problem, it must contain the
            problem type (TS, Class. or Reg.), input columns, target column,
            and some attributes that each problem type requires.

        Returns:
        -------
        response: Dictionary
            A dictionary that contains needed parameters for the best model found from
            problem configuration. Each element shows (name, type, default value)
            for each parameters.
            For example, in a Time Series problem where ARIMA
            is decided to be the best model. It will return 

            {
                'model_name': 'ARIMA',
                'params': [
                    {'name': 'p', 'type': 'int', 'default': 7},
                    {'name': 'd', 'type': 'int', 'default': 3}, 
                    ...
                ]
            }
        """
        # TODO: Get problem from request
        abstract_factory = Factory()
        problem = request.problem_type

        self.factory = abstract_factory.get_factory(problem)
        # df = self.localfile.read_dataframe(request.dataset)
        # TODO: Configurate problem & dataset from request
        dataset = self.factory.create_dataset(data=data)

        problem = self.factory.create_problem(
            label=request.target,
            input_cols=request.input_cols,
            date_col=request.date_col,
            first_config=True,
        )
        # Reset pipeline when there is a change
        if hasattr(self, "pipeline"):
            delattr(self, "pipeline")

        response_dict = {
            "model_name": problem.best_model.model_name, "params": problem.best_model.h}
        return response_dict
