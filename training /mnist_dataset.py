import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from norse.torch.functional.encode import poisson_encode
import numpy as np
from brian2 import *


class MnistDataset(Dataset):

    def __init__(self, train_mode: bool, transform: transforms.Compose = None, labels_to_use: list = None, binary: bool = False):
        mnist_set = datasets.MNIST(root='./data', train=train_mode, download=True)
        self.data = mnist_set.data
        self.targets = mnist_set.targets
        self.tranform = transform
        self.binary = binary
        if labels_to_use is not None:
            self.set_update_dataset(labels_to_use)

    def set_update_dataset(self, labels_to_use: list):
        indices = torch.squeeze(torch.cat([torch.argwhere(self.targets == l) for l in labels_to_use], dim=0))
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        target = self.targets[index]
        data = self.normalize(data)
        data = self.to_binary(data) if self.binary else data
        data = self.pixels_to_firing_rate(data)
        return data, target, index

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data)) #TBD: consider using torch transformers instead (Read

    def to_binary(self, data: torch.Tensor, th: int = 128):
        return (data > th).to(dtype=float)

    def pixels_to_firing_rate(self, data: np.ndarray, seq_length: int = 100, f_max: int = 100, dt_sec: float = 0.001) -> np.ndarray:
        encoded_data = poisson_encode(data, seq_length=seq_length, f_max=f_max, dt=dt_sec)
        # Calculate firing rates for each pixel:
        firing_rates = encoded_data.numpy().mean(axis=0) / dt_sec  # Spikes per second (Hz)
        return firing_rates.flatten()

    def firing_rate_to_poisson_group(self, num_neurons: int, firing_rates):
        # Define the PoissonGroup with firing rates
        firing_rates = firing_rates.flatten()
        poisson_group = PoissonGroup(num_neurons, rates=firing_rates * Hz)
        return poisson_group


