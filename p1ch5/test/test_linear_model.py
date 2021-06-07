import torch

from p1ch5.linear_model import LinearModelData, HyperParameters, training_loop
from p1ch5.render import render_temp_measurements


def test_linear_model_learning_overtraining():
    # In this example, the learning rate is 1e-02 which is too large and results
    # in the loss becoming infinite (inf)

    # Set hyper parameters
    learning_rate = torch.tensor([1e-2], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    parameters = torch.tensor([1.0, 0.0])  # [Weights, Bias]

    data = LinearModelData(temp_celsius=t_c,
                           temp_unknown=t_u)

    # Plot the inputs
    render_temp_measurements(data=data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=100, model_data=data, hyper_parameters=hyper_parameters,
                                     parameters=parameters, print_step=1, print_parameters=True)


def test_linear_model_learning_stalling():
    # In this example, the learning rate is 1e-04 which is not too large.
    # However the results stall with loss around 29.02
    # We could improve this will normalizing

    # Set hyper parameters
    learning_rate = torch.tensor([1e-4], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    parameters = torch.tensor([1.0, 0.0])  # [Weights, Bias]

    data = LinearModelData(temp_celsius=t_c,
                           temp_unknown=t_u)

    # Plot the inputs
    render_temp_measurements(data=data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=5_000, model_data=data, hyper_parameters=hyper_parameters,
                                     parameters=parameters, print_step=500, print_parameters=True)


def test_linear_model_learning_easy_norm():
    # In this example, the learning rate is 1e-04 which is not too large.
    # However the results stall with loss around 29.02
    # We could improve this will normalizing the unknown temperatures in a very simple
    # way by multiplying with 0.1

    # Set hyper parameters
    learning_rate = torch.tensor([1e-2], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    parameters = torch.tensor([1.0, 0.0])  # [Weights, Bias]

    # Normalize the unknown temperatures to be closer in scale to t_c
    t_u_normalized = t_u * 0.1

    data = LinearModelData(temp_celsius=t_c,
                           temp_unknown=t_u_normalized)

    # Plot the inputs
    render_temp_measurements(data=data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=5_000, model_data=data, hyper_parameters=hyper_parameters,
                                     parameters=parameters, print_step=500, print_parameters=True)
