from brian2 import *
import numpy as np

# Example dataset: 100 images, 28x28 pixels each (flattened to 784)
num_images = 100
image_size = 28 * 28
images = np.random.rand(num_images, image_size)  # Replace with your dataset

# Parameters
input_neurons = image_size
hidden_neurons = 100
output_neurons = 10
simulation_time = 100 * ms  # Time to present each image

# Input layer
input_layer = NeuronGroup(input_neurons, 'rates : Hz', threshold='rand() < rates*dt', method='linear')

# Hidden layer
hidden_layer = NeuronGroup(hidden_neurons, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0')

# Output layer
output_layer = NeuronGroup(output_neurons, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0')

# Synapses
input_to_hidden = Synapses(input_layer, hidden_layer, model='w : 1', on_pre='v_post += w')
input_to_hidden.connect(p=0.1)
input_to_hidden.w = 'rand()'

hidden_to_output = Synapses(hidden_layer, output_layer, model='w : 1', on_pre='v_post += w')
hidden_to_output.connect(p=0.1)
hidden_to_output.w = 'rand()'

# Monitors
spike_mon_input = SpikeMonitor(input_layer)
spike_mon_hidden = SpikeMonitor(hidden_layer)
spike_mon_output = SpikeMonitor(output_layer)

# Training loop
for i, image in enumerate(images):
    print(f"Training on image {i + 1}/{num_images}")

    # Set input rates based on image pixel intensities
    input_layer.rates = image * 50 * Hz  # Scale pixel values to firing rates

    # Simulate
    run(simulation_time)

    # Convert spike counts to NumPy arrays
    input_counts = np.array(spike_mon_input.count)  # Shape: (784,)
    hidden_counts = np.array(spike_mon_hidden.count)  # Shape: (100,)

    # Compute weight updates
    delta_w = (input_counts[:, np.newaxis] @ hidden_counts[np.newaxis, :]).flatten()

    # Update weights
    input_to_hidden.w[:] += 0.01 * delta_w

# Analyze results
plot(spike_mon_output.t / ms, spike_mon_output.i, '.')
xlabel('Time (ms)')
ylabel('Neuron index')
show()