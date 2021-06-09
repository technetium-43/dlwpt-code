import torch
from torch import optim
from typing import NamedTuple
from p1ch5.model_data import ModelData, HyperParameters


def linear_model(model_data: ModelData, parameters: torch.tensor) -> torch.tensor:
    # m = Linear model
    weights = parameters[0]
    bias = parameters[1]

    return weights * model_data.temp_unknown + bias


def loss_error_squared(y_pred: torch.tensor, model_data: ModelData) -> torch.tensor:
    return ((y_pred - model_data.temp_celsius) ** 2).mean()


def training_loop(n_epochs: int, optimizer: torch.optim,
                  training_data: ModelData, parameters: torch.tensor,
                  validation_data: ModelData = None,
                  print_step: int = 500, print_parameters: bool = True) -> torch.tensor:
    for epoch in range(n_epochs):

        # IMPORTANT reset leaves of derivative graph to zero
        optimizer.zero_grad()

        # Forward Pass
        training_temperature_prediction = linear_model(model_data=training_data, parameters=parameters)

        # Calculate Loss
        training_loss = loss_error_squared(y_pred=training_temperature_prediction, model_data=training_data)

        # Calculate forward pass and loss for training data
        if validation_data is not None:
            with torch.no_grad():
                # Disable torch.autograd with torch.no_grade() context manager
                # https://pytorch.org/docs/stable/generated/torch.no_grad.html
                validation_temperature_prediction = linear_model(model_data=validation_data, parameters=parameters)
                validation_loss = loss_error_squared(y_pred=validation_temperature_prediction,
                                                     model_data=validation_data)
                # assert validation_loss.requires_grad == False  # Check that the output requires _grad args are foced to False
        else:
            validation_temperature_prediction = -1
            validation_loss = -1

        # Calculate Gradient of the Loss function using auto_grad
        # https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients
        training_loss.backward()

        # Use the pre-built optimizer to update the parameters for learning using gradient descent
        # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        optimizer.step()

        # Print the results
        if epoch == 0:
            print('\n')
        if not print_parameters and epoch % print_step == 0:
            print(
                f"Epoch: {epoch}, Training loss: {float(training_loss):.4f}, Validation loss: {float(validation_loss):.4f}")
        elif print_parameters and epoch % print_step == 0:
            print(
                f"Epoch: {epoch}, Training loss: {float(training_loss):.4f}, Validation loss: {float(validation_loss):.4f}"
                f"\n\tWeights: {float(parameters[0]):.4f}, Bias: {float(parameters[1]):.4f}"
                f"\n\tLoss Gradient: {parameters.grad}")

    return parameters
