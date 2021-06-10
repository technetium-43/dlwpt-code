from matplotlib import pyplot as plt
from model_data import ModelData, ModelResults
import torch


def render_results(results: ModelResults = None):
    # Create figure with axes
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    results_ax = axes

    # Plot the training and validation results and the original data
    # Original data
    temperature_unknown = results.model_data.temp_unknown.detach().clone()
    temperature_celsius = results.model_data.temp_celsius.detach().clone()
    results_ax.scatter(temperature_unknown, temperature_celsius, marker="x", color='orange', label='All input data')

    # Training Prediction
    temperature_unknown_training = results.training_data.temp_unknown.detach().clone()  # clone and detach to disconnect from gradient graph
    temperature_unknown_training, indices_unknown = torch.sort(temperature_unknown_training)
    temperature_celsius_training_prediction = results.temperature_celsius_training_prediction.detach().clone()
    temperature_celsius_training_prediction, indices_celsius_prediction = torch.sort(temperature_celsius_training_prediction)
    results_ax.plot(temperature_unknown_training, temperature_celsius_training_prediction, marker="o", color='red', label='Training prediction')

    # Validation Prediction
    temperature_unknown_validation = results.validation_data.temp_unknown.detach().clone()
    temperature_unknown_validation, indices_unknown_validation = torch.sort(temperature_unknown_validation)
    temperature_celsius_validation_prediction = results.temperature_celsius_validation_prediction.detach().clone()
    temperature_celsius_validation_prediction, indices_celsius_validation = torch.sort(temperature_celsius_validation_prediction)
    results_ax.plot(temperature_unknown_validation, temperature_celsius_validation_prediction, marker="o", color='green', label='Validation prediction')

    # Labels
    fig.suptitle("Prediction Results", fontsize=12)
    results_ax.legend()
    results_ax.set_ylabel("Temperature (Celsius)")
    results_ax.set_xlabel("Temperature (Unknown)")

    plt.show()



def render_scatter_input_data(data: ModelData):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 11))

    temperature_celsius_ax = axes[0]
    temperature_unknown_ax = axes[1]
    merged_ax = axes[2]

    # Prepare data for plotting
    count = data.temp_celsius.size()[0]
    observations = torch.arange(0, count)
    temp_celsius, indices = torch.sort(data.temp_celsius)
    temp_unknown, indices_2 = torch.sort(data.temp_unknown)

    # Create scatter plots
    temperature_celsius_ax.scatter(observations, temp_celsius)
    temperature_unknown_ax.scatter(observations, temp_unknown)
    merged_ax.scatter(temp_unknown, temp_celsius)

    # Labels
    temperature_celsius_ax.set_ylabel("Temperature (Celsius)")
    temperature_celsius_ax.set_xlabel("Measurement")
    temperature_unknown_ax.set_ylabel("Temperature (Unknown)")
    temperature_unknown_ax.set_xlabel("Measurement")
    fig.suptitle("Input Data", fontsize=12)
    merged_ax.set_ylabel("Temperature (Celsius)")
    merged_ax.set_xlabel("Temperature (Unknown)")
    plt.show()