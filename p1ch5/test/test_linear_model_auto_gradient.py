import torch
from p1ch5.model_data import ModelData, HyperParameters
from p1ch5.linear_model_auto_gradient import training_loop
from p1ch5.render import render_scatter_input_data


def test_linear_model():
    # In this example we use Torch's auto gradient graph tracking

    # Set hyper parameters
    learning_rate = torch.tensor([1e-2], dtype=torch.float)
    hyper_parameters = HyperParameters(learning_rate=learning_rate)

    # Organize the model inputs
    temperature_celsius = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    temperature_unknown = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    t_u_normalized = 0.1 * temperature_unknown
    data = ModelData(temp_celsius=temperature_celsius,
                           temp_unknown=t_u_normalized)

    # Model parameters - use requires_grad for parameter auto-gradient
    # https://pytorch.org/docs/stable/autograd.html?highlight=requires_grad#torch.Tensor.requires_grad
    parameters = torch.tensor([1.0, 0.0], requires_grad=True)  # [Weights, Bias]

    # Plot the inputs
    render_scatter_input_data(data=data)

    # Run the training loop
    final_parameters = training_loop(n_epochs=5_000, training_data=data, hyper_parameters=hyper_parameters,
                                     parameters=parameters, print_parameters=True)
    print(f"Final Parameters [Weights, Bias]: {final_parameters}")
