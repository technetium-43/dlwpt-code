from typing import NamedTuple
import torch


class ModelData(NamedTuple):
    temp_celsius: torch.tensor
    temp_unknown: torch.tensor


class HyperParameters(NamedTuple):
    learning_rate: torch.tensor
