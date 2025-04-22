import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Function to encode pixel values into spikes (rate coding)
def rate_coding(image, num_timesteps=20):
    """
    Encode an image into a spike train using rate coding.
    Each pixel intensity determines the spiking probability.

    Args:
        image (torch.Tensor): The input image (1 x H x W) normalized to [0, 1].
        num_timesteps (int): The number of timesteps for the spike train.

    Returns:
        torch.Tensor: A binary spike train (num_timesteps x H x W).
    """
    image = image.squeeze()  # [H, W]
    spike_trains = torch.rand((num_timesteps, *image.shape)) < image  # Binary spikes based on pixel intensity
    return spike_trains.float()

# Example: Encode one image
image, label = mnist_dataset[101]
spike_train = rate_coding(image, num_timesteps=20)

print(f"Label: {label}")
print(f"Spike Train Shape: {spike_train.shape}")  # (20, 28,

'''plt.figure()
for i in range(20):
    tmp = spike_train[i].detach().cpu().numpy()
    plt.subplot(4,5,i+1)
    plt.imshow(tmp, cmap='gray')
plt.show()'''

plt.figure()
for i in range(28):
    tmp = spike_train[:,13, i].detach().cpu().numpy()
    plt.subplot(7,4,i+1)
    plt.plot(tmp)
plt.show()


# Calculate the firing rate (average spikes) as intensity (grayscale)
firing_rate = spike_train.mean(dim=0)  # Average over time for each pixel

# Plot both images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image (pixel intensity values)
axes[0].imshow(image.squeeze(), cmap='gray', interpolation='nearest')  # Display image in grayscale (intensity)
axes[0].set_title(f"Original Image - Label: {label}")
axes[0].axis('off')

# Plot firing rate (grayscale)
axes[1].imshow(firing_rate, cmap='gray', interpolation='nearest')  # Firing rate as intensity
axes[1].set_title("Firing Rate (Grayscale)")
axes[1].axis('off')

plt.show()
