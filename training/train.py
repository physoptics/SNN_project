import argparse
from argparse import Namespace

from norse.task.mnist import train
from sympy.combinatorics.galois import S4xC2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_dataset import MnistDataset
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as T


if __name__ == "__main__":

    num_epochs = 1
    batch_size = 1
    num_workers = 0
    device = 'cuda'
    labels_to_use = [0, 1]

    # reduce MNIST size from 28X28 to 9X9
    transform = T.Compose([
    T.Resize((9, 9), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor()
        ])
    train_dataset = MnistDataset(train_mode=True, transform=transform, labels_to_use=labels_to_use)
    transform_val = transforms.Compose([transforms.ToTensor()])
    validation_dataset = MnistDataset(train_mode=False, transform=transform_val, labels_to_use=labels_to_use)

    dataloader_params = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True, 'num_workers': num_workers}
    train_loader = DataLoader(train_dataset, **dataloader_params)
    validation_loader = DataLoader(validation_dataset, **dataloader_params)

    # Parameters used in the excitatory equation:
    taue = 20 * ms  # time constant that describes the behavior of the excitatory neuron. TBD: consider using 100 ms instead, according to fncom-09-00099.pdf it's supposed to increase classification accuracy
    v_rest_e = -65 * mV  # reset voltage the excitatory neuron is reset to after reaching Vth
    Vth_e = -52 * mV  # membrane threshold (excitatory) - when the neuron membrane crosses its membrane threshold, the neuron fires and it membrane potential is reset to:
    # ge - conductance of excitatory synapses

    neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (taue)  : volt
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-100.*mV-v)                          : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                  : 1
            '''

    # Parameters used in the inhibitory equation:
    taui = 10 * ms  # time constant that describes the behavior of the inhibitory neuron
    v_rest_i = -60 * mV  # reset voltage the inhibitory neuron is reset to after reaching Vth
    Vth_i = -40 * mV  # membrane threshold (inhibitory) - when the neuron membrane crosses its membrane threshold, the neuron fires and it membrane potential is reset to:
    # gi - conductance of inhibitory synapses

    neuron_eqs_i = '''
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (taui)  : volt
            I_synE = ge * nS *         -v                           : amp
            I_synI = gi * nS * (-85.*mV-v)                          : amp
            dge/dt = -ge/(1.0*ms)                                   : 1
            dgi/dt = -gi/(2.0*ms)                                  : 1
            '''

    # input layer - excitatory neurons layer:
    num_of_neurons_e = 500
    neurons_e = NeuronGroup(num_of_neurons_e, neuron_eqs_e, threshold='v>Vth_e', reset='v = v_rest_e',
                            method='euler')  # TBD: euler? maybe some other method? check if 'exact' fits better
    # hidden_layer - inhibitory neurons layer:
    num_of_neurons_i = num_of_neurons_e
    neurons_i = NeuronGroup(num_of_neurons_i, neuron_eqs_i, threshold='v>Vth_i', reset='v = v_rest_i',
                            method='euler')  # TBD: euler? maybe some other method? check if 'exact' fits better
    # output layer:
    n_output = 2
    output_layer = NeuronGroup(n_output, 'dv/dt = -v / (10*ms) : 1', threshold='v > 1', reset='v = 0', method='exact')

    for e in range(num_epochs):
        for batch_ind, (data, targets, ids) in enumerate(train_loader):
            # data, targets = [d.to(device) for d in [data, targets]]
            duration = 100 * ms
            firing_rates = train_dataset.pixels_to_firing_rate(data)
            poisson_input = train_dataset.firing_rate_to_poisson_group(firing_rates = firing_rates, num_neurons = 784)

            # run simulation:
            run(duration)

            # connect the chain: poisson_input -> excitatory neurons -> inhibatory neurons:

            #s1 - STDP parameters:
            taupre = 20 * ms
            taupost = taupre
            gmax = .01
            dApre = .01
            dApost = -dApre * taupre / taupost * 1.05
            dApost *= gmax
            dApre *= gmax

            s1 = Synapses(poisson_input, neurons_e,
                  '''w : 1
                     dApre/dt = -Apre / taupre : 1 (event-driven)
                     dApost/dt = -Apost / taupost : 1 (event-driven)''',
                  on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
                  on_post='''Apost += dApost
                    w = clip(w + Apre, 0, gmax)''',
                          )

            #connect the firing rates to the excitatory neuron group in a one-to-one fashion:
            s1.connect(i = 'j') #one-to-one connection
            s1.w = 'rand() * gmax'
            mon = StateMonitor(s1, 'w', record=[0, 1])
            s_mon = SpikeMonitor(poisson_input)

            #connect the excitatory neuron group to the inhibitory neuron group in a one to one fashion:
            s2 = Synapses(neurons_e, neurons_i, on_pre='ge += we') #on_pre - what happens when a presynaptic neuron spikes.
            s2.connect(i = 'j') #one-to-one connection

            #implement lateral inhibition (each inhibitory neuron is connected to all excitatory neurons except for the one it recieved a connection from:
            s3 = Synapses(neurons_i, neurons_e, on_pre='gi += wi') #on_pre - what happens when a presynaptic neuron spikes.
            s3.connect(condition = 'i != j')

            s4 = Synapses(neurons_e, output_layer, on_pre='ge += we')
            s4.connect()

            # Monitor spikes from output layer:
            output_spike_monitor = SpikeMonitor(output_layer)

            # Compute spike rates:
            spike_counts = torch.tensor([output_spike_monitor.count[i] for i in range(n_output)])

            #convert spike counts to probabilities using softmax:
            probabilities = T.softmax(spike_counts.float(), dim = 0)

            print("spike counts:", spike_counts.numpy())
            print("Probabilities", probabilities.numpy())

