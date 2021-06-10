from typing import NamedTuple
import torch


class ModelData(NamedTuple):
    temp_celsius: torch.tensor
    temp_unknown: torch.tensor


class HyperParameters(NamedTuple):
    learning_rate: torch.tensor


class ModelResults(NamedTuple):
    model_data: ModelData
    training_data: ModelData
    validation_data: ModelData
    temperature_celsius_training_prediction: torch.tensor
    temperature_celsius_validation_prediction: torch.tensor
