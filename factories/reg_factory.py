from forecasting_module.dataset.reg_dataset import RegressionDataset
from forecasting_module.factories.base_factory import BaseFactory
from forecasting_module.pipelines.reg_pipeline import RegressionPipeline
from forecasting_module.problems.regression import RegressionProblem


class RegressionFactory(BaseFactory):
    def create_pipeline(self, **kwargs) -> RegressionPipeline:
        return RegressionPipeline(**kwargs)

    def create_problem(self, **kwargs) -> RegressionProblem:
        return RegressionProblem(**kwargs)

    def create_dataset(self, **kwargs) -> RegressionDataset:
        return RegressionDataset(**kwargs)
