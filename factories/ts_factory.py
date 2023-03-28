from forecasting_module.dataset.ts_dataset import TimeSeriesDataset
from forecasting_module.factories.base_factory import BaseFactory
from forecasting_module.pipelines.ts_pipeline import TimeSeriesPipeline
from forecasting_module.problems.time_series import TimeSeriesProblem


class TimeSeriesFactory(BaseFactory):
    def create_pipeline(self, **kwargs) -> TimeSeriesPipeline:
        return TimeSeriesPipeline(**kwargs)

    def create_problem(self, **kwargs) -> TimeSeriesProblem:
        return TimeSeriesProblem(**kwargs)

    def create_dataset(self, **kwargs) -> TimeSeriesDataset:
        return TimeSeriesDataset(**kwargs)
