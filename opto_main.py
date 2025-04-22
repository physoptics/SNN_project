import torch
from brian2 import *
from torchvision import datasets, transforms
from norse.torch.functional.encode import poisson_encode
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Parameters
seq_length = 100  # Duration of encoding in ms
f_max = 100  # Maximum firing rate in Hz
dt = 0.001  # Time step in seconds

# Select an image and its label
#image, label = mnist_dataset[101]
#print("Original shape of image from dataset:", image.shape)  # Debugging

# Ensure image is in the correct shape (28x28)
image = image.squeeze()  # Removes singleton dimension (1, 28, 28) -> (28, 28)
#print("Shape of image after squeeze:", image.shape)

# Encode image into Poisson spikes
encoded_data = poisson_encode(image, seq_length=seq_length, f_max=f_max, dt=dt)
# print("Shape of encoded_data:", encoded_data.shape)  # Debugging

# Visualize the Poisson-encoded data
# Sum over time axis (axis=1) and reshape to (28, 28) for visualization
plt.imshow(encoded_data.numpy().sum(axis=0).reshape(28, 28), cmap="gray")
plt.title(f"Poisson Encoded Image for Label: {label}")
plt.colorbar()
plt.show()

# Calculate firing rates for each pixel
seq_length = encoded_data.shape[1]  # Time steps
simulation_duration = seq_length * dt * second  # Total simulation time

# Flatten the 2D image to 1D (28x28 -> 784 neurons) and calculate firing rates
firing_rates = encoded_data.numpy().mean(axis=0) / dt  # Spikes per second (Hz)
print("Shape of firing_rates:", firing_rates.shape)  # Debugging

# Define the PoissonGroup with firing rates
num_neurons = firing_rates.size  # Total number of pixels (784 for MNIST)
firing_rates = firing_rates.flatten()
poisson_group = PoissonGroup(num_neurons, rates=firing_rates * Hz)

# Record spikes
spike_monitor = SpikeMonitor(poisson_group)

# Run the simulation
run(simulation_duration)

# Plot spikes
plt.figure(figsize=(10, 6))

# Plot spike raster (neuron index vs. spike time)
plt.plot(spike_monitor.t / ms, spike_monitor.i, ".k", markersize=1)
plt.title("Spike Raster Plot")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron Index")
plt.show()

# Run the simulation
run(simulation_duration)
