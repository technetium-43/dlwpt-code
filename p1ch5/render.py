from matplotlib import pyplot as plt
from model_data import ModelData
import torch


def render_temp_measurements(data: ModelData):
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
    merged_ax.set_ylabel("Temperature (Celsius)")
    merged_ax.set_xlabel("Temperature (Unknown)")
    plt.show()
