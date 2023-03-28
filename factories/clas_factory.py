from forecasting_module.dataset.clas_dataset import ClassificationDataset
from forecasting_module.factories.base_factory import BaseFactory
from forecasting_module.pipelines.clas_pipeline import ClassificationPipeline
from forecasting_module.problems.classification import ClassificationProblem


class ClassificationFactory(BaseFactory):
    def create_pipeline(self, **kwargs) -> ClassificationPipeline:
        return ClassificationPipeline(**kwargs)

    def create_problem(self, **kwargs) -> ClassificationProblem:
        return ClassificationProblem(**kwargs)

    def create_dataset(self, **kwargs) -> ClassificationDataset:
        return ClassificationDataset(**kwargs)
