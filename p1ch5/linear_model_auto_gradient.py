from typing import NamedTuple
from dataclasses import dataclass

import torch


class LinearModelData(NamedTuple):
    temp_celsius: torch.tensor
    temp_unknown: torch.tensor


class HyperParameters(NamedTuple):
    learning_rate: torch.tensor


def linear_model(model_data: LinearModelData, parameters: torch.tensor) -> torch.tensor:
    # m = Linear model
    weights = parameters[0]
    bias = parameters[1]

    return weights * model_data.temp_unknown + bias


def loss_error_squared(y_pred: torch.tensor, model_data: LinearModelData) -> torch.tensor:
    return ((y_pred - model_data.temp_celsius) ** 2).mean()


def training_loop(n_epochs: int, model_data: LinearModelData, hyper_parameters: HyperParameters,
                  parameters: torch.tensor, print_parameters: bool = True) -> torch.tensor:
    for epoch in range(n_epochs):

        # IMPORTANT reset leaves of derivative graph to zero
        if parameters.grad is not None:
            parameters.grad.zero_()

        # Forward Pass
        temperature_prediction = linear_model(model_data=model_data, parameters=parameters)

        # Calculate Loss
        loss = loss_error_squared(y_pred=temperature_prediction, model_data=model_data)

        # Calculate Gradient of the Loss function using auto_grad
        # https: // pytorch.org / docs / stable / autograd.html  # module-torch.autograd
        loss.backward()

        # Update parameters for learning
        with torch.no_grad():
            parameters -= hyper_parameters.learning_rate * parameters.grad

        # Print the results
        if epoch == 0:
            print('\n')
        if not print_parameters and epoch % 500 == 0:
            print(f"Epoch: {epoch}, loss: {float(loss):.4f}")
        elif print_parameters and epoch % 500 == 0:
            print(f"Epoch: {epoch}, loss: {float(loss):.4f}\n\tWeights: {float(parameters[0]):.4f}, "
                  f"Bias: {float(parameters[1]):.4f}\n\tLoss Gradient: {parameters.grad}")

    return parameters
