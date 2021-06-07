from matplotlib import pyplot as plt
from linear_model import LinearModelData
import torch
import numpy as np


def render_temp_measurements(data: LinearModelData):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 11))

    celcius_ax = axes[0]
    unkown_ax = axes[1]
    merged_ax = axes[2]

    # Prepare data for plotting
    count = data.temp_celsius.size()[0]
    observations = torch.arange(0, count)
    temp_celsius, indices = torch.sort(data.temp_celsius)
    temp_unknown, indices_2 = torch.sort(data.temp_unknown)

    # Create scatter plots
    celcius_ax.scatter(observations, temp_celsius)
    unkown_ax.scatter(observations, temp_unknown)
    merged_ax.scatter(temp_unknown, temp_celsius)

    # Labels
    celcius_ax.set_ylabel("Temperature (Celsius)")
    celcius_ax.set_xlabel("Measurement")
    unkown_ax.set_ylabel("Temperature (Unknown)")
    unkown_ax.set_xlabel("Measurement")
    merged_ax.set_ylabel("Temperature (Celsius)")
    merged_ax.set_xlabel("Temperature (Unknown)")
    plt.show()
