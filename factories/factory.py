from api_module.model.problem_types import ProblemType
from forecasting_module.factories.base_factory import BaseFactory
from forecasting_module.factories.clas_factory import ClassificationFactory
from forecasting_module.factories.reg_factory import RegressionFactory
from forecasting_module.factories.ts_factory import TimeSeriesFactory
from loguru import logger


class Factory:
    def get_factory(self, problem_type) -> BaseFactory:
        if problem_type == ProblemType.TIMESERIES:
            return TimeSeriesFactory()
        elif problem_type == ProblemType.CLASSIFICATION:
            return ClassificationFactory()
        elif problem_type == ProblemType.REGRESSION:
            return RegressionFactory()
        else:
            logger.info("[FORECASTING MODULE] There is no problem with name {}".format(problem_type))
