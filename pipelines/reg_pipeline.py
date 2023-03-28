from typing import Union
from forecasting_module.models.base_model import BaseModel
from forecasting_module.models.framework_utils.wrapper import ModelFrameworkService
from forecasting_module.pipelines.base_pipeline import BasePipeline
from forecasting_module.problems.regression import RegressionProblem
from reeco_ml_preprocessing.process_numerical.standard_scaler import StandardScaler
from reeco_ml_preprocessing.convert_data_type.pd_np_transform import PandasNumpyTransformer
from reeco_ml_preprocessing.preprocess_pipeline import PreprocessPipeline
from reeco_ml_preprocessing.time_series.ts_imputer import TimeSeriesImputer
from reeco_ml_preprocessing.time_series.cleaner import TimeSeriesCleaner


class RegressionPipeline(BasePipeline):
    def __init__(self, problem: RegressionProblem, model: Union[BaseModel, ModelFrameworkService] = None, **kwargs):
        self.impute_method = "mean"
        self.model = model
        self.problem = problem
        if "pipeline_id" in kwargs:
            self.pipeline_id = kwargs["pipeline_id"]
        else:
            self.pipeline_id = "0"

    def make_pipeline(self):
        self.transformer = PreprocessPipeline([
            TimeSeriesImputer(method=self.impute_method),
            PandasNumpyTransformer()
        ])

    def fit(self, source):
        X, y = source
        self.make_pipeline()
        X = self.transformer.fit_transform(X)
        y = self.transformer.transform(y)
        return self.model.fit(X, y, self.problem, self.pipeline_id)

    def predict(self, X):
        X = self.transformer.fit_transform(X)
        y = self.model.predict(X)
        y = self.transformer.inverse_transform(y)
        return y

    def validate(self, source):
        X, y = source
        X_trans = self.transformer.transform(X)
        y_pred = self.model.predict(X_trans)
        # Concat actual value and predicted value
        y = y.to_frame()
        y.columns = ["actual"]
        y["predicted"] = y_pred
        y.dropna(inplace=True)
        return y

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
        self.transformer.save(path=f"ckpts/preprocessor_{self.pipeline_id}.joblib")
