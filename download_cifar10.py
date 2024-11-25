import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

# Directory to save real images
real_image_dir = "/Users/darshgondalia/Desktop/CS 682/Final Project/real_images_10"
os.makedirs(real_image_dir, exist_ok=True)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
cifar10_dataset = datasets.CIFAR10(
    root='./data',
    train=False,  # Use test set
    download=True,
    transform=transform
)

# DataLoader
real_loader = DataLoader(cifar10_dataset, batch_size=10, shuffle=False)

num_images = 100
# Save 10 real images
image_counter = 0
for images, _ in tqdm(real_loader, desc="Saving real images"):
    for i in range(images.size(0)):
        image = images[i]
        image_save_path = os.path.join(real_image_dir, f"real_{image_counter:05d}.png")
        torchvision.utils.save_image(
            image,
            image_save_path,
            normalize=True,
            value_range=(-1, 1)
        )
        image_counter += 1
        if image_counter >= num_images:
            break
    if image_counter >= num_images:
        break