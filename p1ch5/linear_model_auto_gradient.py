from typing import NamedTuple, Optional
import torch

from p1ch5.model_data import ModelData, HyperParameters


def linear_model(model_data: ModelData, parameters: torch.tensor) -> torch.tensor:
    # m = Linear model
    weights = parameters[0]
    bias = parameters[1]

    return weights * model_data.temp_unknown + bias


def loss_error_squared(y_pred: torch.tensor, model_data: ModelData) -> torch.tensor:
    return ((y_pred - model_data.temp_celsius) ** 2).mean()


def training_loop(n_epochs: int, training_data: ModelData,
                  hyper_parameters: HyperParameters, parameters: torch.tensor,
                  print_parameters: bool = True) -> torch.tensor:
    for epoch in range(n_epochs):

        # IMPORTANT reset leaves of derivative graph to zero
        if parameters.grad is not None:
            parameters.grad.zero_()

        # Forward Pass
        temperature_prediction = linear_model(model_data=training_data, parameters=parameters)

        # Calculate Loss
        loss = loss_error_squared(y_pred=temperature_prediction, model_data=training_data)

        # Calculate Gradient of the Loss function using auto_grad
        # https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients
        loss.backward()

        # Update parameters for learning using gradient descent
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
