from torch.utils.data import DataLoader
from brian2 import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset
from models.brain_model import BrainModel

if __name__ == "__main__":

    num_epochs = 1
    batch_size = 1
    num_workers = 0
    device = 'cpu'
    labels_to_use = [0, 1]
    input_size = 784 #TBD 

    train_dataset = MnistDataset(train_mode=True, labels_to_use=labels_to_use)
    validation_dataset = MnistDataset(train_mode=False, labels_to_use=labels_to_use)

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True, 'num_workers': num_workers}
    train_loader = DataLoader(train_dataset, **dataloader_params)
    validation_loader = DataLoader(validation_dataset, **dataloader_params)

    #Intialize model:
    model = BrainModel(input_size = input_size)

    for e in range(num_epochs):
        for batch_ind, (data, targets, ids) in enumerate(train_loader):
            # data, targets = [d.to(device) for d in [data, targets]]
            duration = 1000 * ms
            # Convert image pixels to firing rates
            firing_rates = data.numpy()

            # Update the rates for the existing Poisson input neurons in the model
            model.set_input_rates(firing_rates.reshape(-1).astype(float))

            print("Min input rate:", min(firing_rates.flatten()))
            print("Max input rate:", max(firing_rates.flatten()))

            # Run the model for the specified duration
            model.run()
            
            #TBD:
            monitors = model.get_monitors()
            # Access output_spike_monitor from the monitors dictionary

            input_spike_monitor = monitors["input_spike_monitor"]
            output_spike_monitor = monitors["output_spike_monitor"]
            excitatory_spike_monitor = monitors["hidden_e_spike_monitor"]
            inhibitory_spike_monitor = monitors["hidden_i_spike_monitor"]

            print("Input spikes:", input_spike_monitor.num_spikes)
            print("Excitatory hidden spikes:", excitatory_spike_monitor.num_spikes)
            print("Inhibitory hidden spikes:", inhibitory_spike_monitor.num_spikes)
            print("Output spikes:", output_spike_monitor.num_spikes)

            # Get spike counts from each output neuron
            n_output = len(model.output_layer)
            spike_counts = torch.tensor([output_spike_monitor.count[i] for i in range(n_output)])

            
            # Duration of the simulation in seconds (e.g., 0.1 s if duration = 100*ms)
            duration_sec = float(duration / second)

            # Compute firing rates (Hz)
            firing_rates_hz = spike_counts.float() / duration_sec

            # Convert spike counts to probabilities using softmax
            probabilities = F.softmax(spike_counts.float(), dim=0)

            print("Spike counts:", spike_counts.numpy())
            print("Probabilities:", probabilities.numpy())

