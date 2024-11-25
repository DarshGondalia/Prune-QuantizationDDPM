import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Subset

save_path = "/Users/darshgondalia/Desktop/CS 682/Final Project/pruned_model/0_3"
# Path to your pruned model
pruned_model_path = os.path.join(save_path, "unet_pruned.pth")

# Load the pruned model
model = torch.load(pruned_model_path)
model.train()  # Set the model to training mode


# Define the transformations (normalize images between -1 and 1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the CIFAR-10 dataset
cifar10_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Randomly select 10,000 indices
total_images = len(cifar10_dataset)
subset_size = 10000
subset_indices = np.random.choice(total_images, subset_size, replace=False)
subset_indices = subset_indices.tolist()

# Create the subset
cifar10_subset = Subset(cifar10_dataset, subset_indices)

# Create a DataLoader
batch_size = 64  # Adjust based on your GPU memory
dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)
# dataloader = DataLoader(cifar10_subset, batch_size=batch_size, shuffle=True)


# Initialize the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=500)

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()


# Set up training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
num_epochs = 1  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        images, _ = batch
        images = images.to(device)

        # Sample random timesteps for each image
        batch_size = images.size(0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        # Add noise to the images
        noise = torch.randn_like(images)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps).sample

        # Calculate loss
        loss = criterion(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    # Save the model after each epoch
    # if epoch%5==0:
    #     torch.save(model.state_dict(), os.path.join(save_path, f"unet_finetuned_epoch_{epoch+1}.pth")) 

# Save the final fine-tuned model
torch.save(model, os.path.join(save_path, "unet_finetuned_full.pth"))
