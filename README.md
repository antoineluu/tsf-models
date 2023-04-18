<div align="center">
<h1> FORECASTING MODULE
</div>


## Description
![Forecast Module](/figures/forecast-module.png)

This module's entrypoints are methods from the ForecastingModule class in (defined in main.py). They provide the user with full pipeline customization for preprocessing, training, hyperparameter tuning. Model Saving for future inference is also implemented.

This module must only be called by the MLService object in /api_module/services/ml_services.py and was not intended to be used as a standalone.

Here we provide a brief description of the standard workflow for the calling of this module's methods for training.

## Workflow

1. `config_problem`: Take `ConfigTrainingRequest` object and sets up the pipeline file with the problem configuration. Final step before training or tuning. Returns a dictionary with `model_name` and `params` for the best configuration given the problem.

2. `tune`: Starts the hyperparameter tuning process based on the tuning request in which a search space is defined for each parameter. The tuning process reiterates the model fitting process (called trial) on the data, re-sampling the model configuration from each parameter from their distribution that is updated at each increment. `tuning_params` must be set while other params follow a strict syntax which we described in the automatic documentation of the DataApp. This methods returns the best pipeline configuration among all the passed trials.

3. `train`: Start Training session based on the input request for the given pipeline. The input data while be automatically separated into training, validation and test datasets according to the `validation size` parameter. This is a high-level method so training objects (dataset, pipeline, model) are all handled by the module. All the necessary information should be included in the request. If `model_params` argument is missing in the request, the method will select parameters from the latest tuning session if there is one. Omitted request params will be set as default. Once the training is complete, the entire pipeline object (including trained weights) is returned along with the result and logs as additional information to characterize the training session.

4. `forecast`: Forecast the immediate future values after the last timesteps of the problem dataset. `transformers` and `onnx_model` are respectively the fitted preprocessor object and model in onnx format.

## Models
* **DilatedCNN**
* **TCN**
* **N-HiTS**
* **GRU**
* **LSTM**
* **Residual LSTM** (not stable)
* **Linear**
* **NLinear**

## Repository
* *callbacks*: The trainer can be served with `callbacks` objects which are called during training or tuning to tweak the training process or retrieve information that is served in real-time to a training manager to monitor metrics.
* *data_access*:
* *dataset*: ts_dataset.py defines a dataset object which does the parting of training, validation, and test sets. The dataset objects are created at the start of train and tune process.
* *factories*: Factory used to create pipelines according to problem type (Forecasting, Regression, Classification)
* *models*: Wrapper class  torch.nn.model objects and default parameters. Each model inherits from BaseModel which sets the common procedures, i.e. training, tuning and export. BaseModel interprets interprets tuning/training requests to form pytorch objects (trainer, dataloader, model).
* *params_gen*:
* *pipelines*: Pipeline objects wrap the preprocessor and the forecasting model to assure streamlined computation. Class methods to fit the pipeline, validate, test, on the data.
* *problems*: Store problem characteristics and serves the best model architecture based on the problem.