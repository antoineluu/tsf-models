from typing import Union
import pandas as pd
from forecasting_module.models.base_model import BaseModel
from forecasting_module.models.framework_utils.wrapper import ModelFrameworkService
from forecasting_module.pipelines.base_pipeline import BasePipeline
from forecasting_module.problems.time_series import TimeSeriesProblem
from reeco_ml_preprocessing.create_static_covariate.lunar import LunarConcater
from reeco_ml_preprocessing.preprocess_pipeline import PreprocessPipeline
from reeco_ml_preprocessing.process_numerical.standard_scaler import StandardScaler
from reeco_ml_preprocessing.process_numerical.minmax_scaler import MinmaxScaler
from reeco_ml_preprocessing.process_numerical.remove_outliers import RemoveOutliers
from reeco_ml_preprocessing.time_series.cleaner import TimeSeriesCleaner
from reeco_ml_preprocessing.time_series.sliding_window import SlidingWindow
from reeco_ml_preprocessing.time_series.ts_imputer import TimeSeriesImputer
from reeco_ml_preprocessing.time_series.angle_mapping import CosSinMapping


class TimeSeriesPipeline(BasePipeline):
    def __init__(self, problem: TimeSeriesProblem, model: Union[BaseModel, ModelFrameworkService] = None, **kwargs):
        # Apply linear interpolation (currently hard coded)
        self.impute_method = "linear"
        self.model = model
        self.problem = problem
        if "pipeline_id" in kwargs:
            self.pipeline_id = kwargs["pipeline_id"]
        else:
            self.pipeline_id = "0"

    def make_pipeline(self, transformers_params=None):

        prefix = [
            TimeSeriesCleaner(
                    input_cols=self.problem.input_cols,
                    date_col=self.problem.date_col,
                    label_col=self.problem.label,
                    sampling_rule=self.problem.sampling_rule,
                    time_lag=self.problem.time_lag,
            ),
            TimeSeriesImputer(method=self.impute_method)
        ]
        suffix = [
            SlidingWindow(
                input_timesteps=self.problem.input_timesteps,
                output_timesteps=self.problem.output_timesteps,
                target=self.problem.label,
            )
        ]
        modules =[]
        module_param_mapping = {
            "RemoveOutliers":{"target":self.problem.label},
            "LunarConcater":{},
            "CosSinMapping":{},
            "StandardScaler":{"target":self.problem.label},
            "MinMaxScaler":{"target":self.problem.label}
        }
        Optional_modules = ["RemoveOutliers", "LunarConcater", "CosSinMapping", "StandardScaler", "MinmaxScaler"]
        for pp_module_string in Optional_modules:
            if pp_module_string in transformers_params.keys():
                modules.append(eval(pp_module_string)(**transformers_params[pp_module_string],**module_param_mapping[pp_module_string]))
        self.transformers = PreprocessPipeline(prefix + modules + suffix)


    def fit(self, train_data, val_data=None):
        X_train, y_train, _ = self.fit_transform(train_data)

        if val_data is None:
            X_val, y_val = None, None
        else:
            X_val, y_val, _ = self.transform(val_data)
        logs = self.model.fit(X_train, y_train, X_val, y_val,
                       pipeline_id=self.pipeline_id, verbose=True)
        return logs

    def transform(self, train_data):
        return self.transformers.transform(train_data)
    
    def fit_transform(self, train_data):
        if not hasattr(self, 'transformers'):
            self.make_pipeline()
        return self.transformers.fit_transform(train_data)

    def tune_model(self, train_data, val_data, params_update, **kwargs):
        X_train, y_train, _ = self.fit_transform(train_data)
        X_val, y_val, _ = self.transform(val_data)
        return self.model.optimize(X_train, y_train, X_val, y_val, self.pipeline_id, params_update, **kwargs)

    def test(self, test_data, rescaled=False, return_loss=False):
        X, y, columns = self.transform(test_data)
        y_pred, loss = self.model.predict(X, y)
        if rescaled:
            X = self.transformers.inverse_transform(X)
            y = self.transformers.inverse_transform(y)
            y_pred = self.transformers.inverse_transform(y_pred)
        if return_loss:
            return y_pred, X, y, columns, loss
        return y_pred, X, y, columns
    
    def validate(self, test_data):

        y_pred, X, y_actual, columns = self.test(
            test_data, rescaled=True, return_loss=False)
        y_actual = y_actual[:, 0, -1].squeeze()
        y_pred = y_pred[:, 0, -1].squeeze()

        indices = pd.date_range(
            start=test_data.index[self.problem.input_timesteps+self.problem.output_timesteps],
            freq=self.problem.sampling_rule,
            periods=X.shape[0],
        )
        # Concat actual value and predicted value
        df = pd.DataFrame(
            {"actual": y_actual, "predicted": y_pred}, index=indices)
        df.dropna(inplace=True)
        return df

    def export(self):
        """
        Export the pipeline for deployment.

        The preprocessor will be exported as a `.joblib` file
        and the model will be exported as `.onnx`.
        """
        self.model.save_model(
            output_path=f"ckpts/model_{self.pipeline_id}.onnx",
            model=self.model.model,
            name=self.model.model_name,
            args=self.model.args if hasattr(self.model, "args") else None,
        )

        self.transformers.save(
            path=f"ckpts/preprocessor_{self.pipeline_id}.joblib")
        