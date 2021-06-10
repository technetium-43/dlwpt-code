import torch
from torch import optim

from p1ch5.model_data import ModelData, HyperParameters, ModelResults
from p1ch5.polynomial_model import training_loop, polynomial_model
from p1ch5.render import render_results, render_scatter_input_data

'''
5.7 Exercise 1 Redefine the model to be w2 * t_u ** 2 + w1 * t_u + b.
a What parts of the training loop, and so on, need to change to accommodate this redefinition? 
b What parts are agnostic to swapping out the model? 
c Is the resulting loss higher or lower after training? 
d Is the actual result better or worse
'''


def test_linear_model():
    # In this example we use Torch's auto gradient graph tracking PLUS we will use the build in optimizer.SGD
    # to calculate the parameter updates by gradient descent.

    # Set hyper parameters
    learning_rate = torch.tensor([1e-4], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    temperature_celsius = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    temperature_unknown = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    temperature_unknown_normalized = 0.1 * temperature_unknown
    training_data = ModelData(temp_celsius=temperature_celsius, temp_unknown=temperature_unknown_normalized)

    # Model parameters - use requires_grad for parameter auto-gradient
    # The model is now a polynomial with two weights m = w2 * t_u ** 2 + w1 * t_u + b.
    # https://pytorch.org/docs/stable/autograd.html?highlight=requires_grad#torch.Tensor.requires_grad
    weights = torch.tensor([1.0, 0.5])
    bias = torch.tensor([0.0])
    parameters = torch.cat((weights, bias), dim=0)
    parameters.requires_grad = True  # We'll use optim.step() to update the parameters with gradient descent

    # Instantiate a gradient descent optimizer
    sgd_optimizer = optim.SGD([parameters], lr=hyper_parameters.learning_rate[0])

    # Plot the inputs
    render_scatter_input_data(data=training_data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=5_001, optimizer=sgd_optimizer, training_data=training_data,
                                     parameters=parameters, print_step=100, print_parameters=False)
    print(f"Final Parameters [Weights, Bias]: {final_parameters}")


def test_linear_model_validation():
    # Set hyper parameters
    learning_rate = torch.tensor([1e-4], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    temperature_celsius = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    temperature_unknown = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    temperature_unknown_normalized = 0.1 * temperature_unknown

    # Split 80:20 the input data to a training and validation set. Randomly sample data into each set using shuffling.
    sample_size = temperature_unknown.shape[0]
    sample_size_validation = int(0.2 * sample_size)
    shuffled_indices = torch.randperm(sample_size)
    training_indices = shuffled_indices[:-sample_size_validation]  # 80% of sample size
    validation_indices = shuffled_indices[-sample_size_validation:]  # 20% of sample size

    training_temperature_celsius = temperature_celsius[training_indices]
    training_temperature_unknown = temperature_unknown[training_indices]
    training_temperature_unknown_normalized = 0.1 * training_temperature_unknown

    validation_temperature_celsius = temperature_celsius[validation_indices]
    validation_temperature_unknown = temperature_unknown[validation_indices]
    validation_temperature_unknown_normalized = 0.1 * validation_temperature_unknown

    training_data = ModelData(temp_celsius=training_temperature_celsius,
                              temp_unknown=training_temperature_unknown_normalized)

    validation_data = ModelData(temp_celsius=validation_temperature_celsius,
                                temp_unknown=validation_temperature_unknown_normalized)

    # Model parameters - use requires_grad for parameter auto-gradient
    # The model is now a polynomial with two weights m = w2 * t_u ** 2 + w1 * t_u + b.
    # https://pytorch.org/docs/stable/autograd.html?highlight=requires_grad#torch.Tensor.requires_grad
    weights = torch.tensor([1.0, 0.5])
    bias = torch.tensor([0.0])
    parameters = torch.cat((weights, bias), dim=0)
    parameters.requires_grad = True  # We'll use optim.step() to update the parameters with gradient descent

    # Instantiate a gradient descent optimizer
    sgd_optimizer = optim.SGD([parameters], lr=hyper_parameters.learning_rate[0])

    # Plot the inputs
    render_scatter_input_data(data=training_data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=5_001, optimizer=sgd_optimizer, training_data=training_data,
                                     parameters=parameters, validation_data=validation_data,
                                     print_step=500, print_parameters=False)
    print(f"\nFinal Parameters [Weights, Bias]: {final_parameters}")

    # Calculate fitted results & render
    model_data = ModelData(temp_celsius=temperature_celsius,
                           temp_unknown=temperature_unknown_normalized)
    training_fit_results = polynomial_model(model_data=training_data, parameters=final_parameters)
    validation_fit_results = polynomial_model(model_data=validation_data, parameters=final_parameters)

    results = ModelResults(model_data=model_data,
                           training_data=training_data,
                           validation_data=validation_data,
                           temperature_celsius_training_prediction=training_fit_results,
                           temperature_celsius_validation_prediction=validation_fit_results)

    render_results(results=results)

    print("Done")
