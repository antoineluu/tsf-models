from forecasting_module.dataset.base_dataset import BaseDataset
from forecasting_module.pipelines.base_pipeline import BasePipeline
from forecasting_module.problems.base import BaseProblem


class BaseFactory:
    def create_pipeline(self) -> BasePipeline:
        raise NotImplementedError

    def create_problem(self) -> BaseProblem:
        raise NotImplementedError

    def create_dataset(self) -> BaseDataset:
        raise NotImplementedError
