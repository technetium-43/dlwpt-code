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


def d_model_d_weights(model_data: LinearModelData) -> torch.tensor:
    # du1/dw = Derivative of Linear Model function with respect to the Weight parameter
    return model_data.temp_unknown


def d_linear_model_d_bias() -> torch.tensor:
    # du1/db = Derivative of Linear Model function with respect to the Bias parameter
    return torch.ones(())


def d_loss_d_linear_model(y_pred: torch.tensor, model_data: LinearModelData) -> torch.tensor:
    # dl/dm = Derivative of loss with respect to the nested function u1
    # u1 = linear model
    return 2 * (y_pred - model_data.temp_celsius)


def gradient_loss(y_pred: torch.tensor, model_data: LinearModelData) -> torch.tensor:
    dl_du1 = d_loss_d_linear_model(y_pred=y_pred, model_data=model_data)
    du1_dw = d_model_d_weights(model_data=model_data)
    du1_db = d_linear_model_d_bias()

    # Sample size
    N = model_data.temp_celsius.size(0)

    # Gradient of Loss with respect to Weights
    dl_dw = ((dl_du1 * du1_dw) / N).sum()

    # Gradient of Loss with respect to Bias
    dl_db = ((dl_du1 * du1_db) / N).sum()

    return torch.stack([dl_dw, dl_db])


def training_loop(n_epochs: int, model_data: LinearModelData, hyper_parameters: HyperParameters,
                  parameters: torch.tensor, print_step: int, print_parameters: bool = True) -> torch.tensor:
    for epoch in range(n_epochs):
        # Forward Pass
        temperature_prediction = linear_model(model_data=model_data, parameters=parameters)

        # Calculate Loss
        loss = loss_error_squared(y_pred=temperature_prediction, model_data=model_data)

        # Calculate Gradient of the Loss function
        loss_gradient = gradient_loss(y_pred=temperature_prediction, model_data=model_data)

        # Update parameters for learning
        parameters -= hyper_parameters.learning_rate * loss_gradient

        # Print the results
        if epoch == 0:
            print('\n')
        if not print_parameters and epoch % print_step == 0:
            print(f"Epoch: {epoch}, loss: {float(loss):.4f}")
        elif print_parameters and epoch % print_step == 0:
            print(f"Epoch: {epoch}, loss: {float(loss):.4f}\n\tWeights: {float(parameters[0]):.4f}, "
                  f"Bias: {float(parameters[1]):.4f}\n\tLoss Gradient: {loss_gradient}")
