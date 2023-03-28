class BasePipeline:
    """
    An abstract pipeline that maps input directly to output with a single call.

    Args:
    -------
        model: BaseModel
            The model used for training and prediction. It can be retrieved via
            `problem.get_best_model()()` (double paranthesis for instantiation).
            For convenience, we takes `model` and `problem` as separated arguments
            instead of taking all in `problem`.
        problem: BaseProblem
            The problem definition.
    """

    def __init__(self, model, problem):
        pass

    def make_pipeline(self):
        """Create pipeline based on choice."""
        raise NotImplementedError

    def fit(self, source):
        """
        Fit data into the pipeline.

        Args:
        -------
            source: Any
                The source data to be worked with. In `TimeSeriesProblem`, source is
                a time series DataFrame since we cannot separate those during transformation.
                In `RegressionProblem` and `ClassificationProblem`, source is a tuple
                of 2 DataFrame representing `X` (predictor) and `y` (target).
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predict without target data.

        Args:
        -------
            X: pd.DataFrame
                Predictor data, as opposed to `fit` and `validate` functions, it does not
                require any composite type of X.

        Returns:
        -------
            y: np.ndarray
                Target predictions, use `np.ndarray` for simplicity.
        """
        raise NotImplementedError

    def validate(self, source):
        """
        Validate model prediction.

        Args:
        -------
            source: Any
                The source data to be worked with. In `TimeSeriesProblem`, source is
                a time series DataFrame since we cannot separate those during transformation.
                In `RegressionProblem` and `ClassificationProblem`, source is a tuple
                of 2 DataFrame representing `X` (predictor) and `y` (target).

        Returns:
        -------
            y: pd.Dataframe
                A DataFrame with `date_col` as index and 2 columns: `actual` and `predicted`.
                The names say its meanings. The DataFrame is used for displaying line plot
                in validation step on UI.
        """
        raise NotImplementedError

    def export(self):
        """"""
