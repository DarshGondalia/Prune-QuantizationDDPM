# import diffusers.models.unets.unet_2d_blocks as models
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
from diffusers import (DDIMPipeline, DDIMScheduler, DDPMPipeline,
                       DDPMScheduler, UNet2DModel)
from diffusers.models.unets.unet_2d_blocks import (DownBlock2D, UNetMidBlock2D,
                                                   UpBlock2D)

# print(dir(models))
## Load DDPM pipeline
pipeline = DDPMPipeline.from_pretrained("./ddpm_cifar10_32")
scheduler = pipeline.scheduler

# Load the pre-trained UNet2DModel
model = pipeline.unet.eval()

# Create example inputs
sample_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 color channels, 32x32 image
timestep = torch.tensor([10], dtype=torch.long)  # Example timestep
example_inputs = {'sample': sample_input, 'timestep': timestep}

# Define the importance criterion (L1 Norm)
importance = tp.importance.MagnitudeImportance()  # L1 Norm

# Identify layers to ignore
ignored_layers = [model.conv_out]

# Exclude specific layers from pruning
p = 0.3
p_string = str(p).replace(".", "_")

# Create the pruner
pruner = tp.pruner.MagnitudePruner(
    model=model,
    example_inputs=example_inputs,
    importance=importance,
    global_pruning=False,
    pruning_ratio=p,  # Prune 20% of the channels
    ignored_layers=ignored_layers,
    channel_groups={},
)


base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
model.zero_grad()
model.eval()

# Apply pruning
for g in pruner.step(interactive=True):
    print(g)
    g.prune()
    
from diffusers.models.resnet import Downsample2D, Upsample2D

for m in model.modules():
    if isinstance(m, (Upsample2D, Downsample2D)):
        m.channels = m.conv.in_channels
        m.out_channels == m.conv.out_channels

macs, params = tp.utils.count_ops_and_params(model, example_inputs)
print(model)
print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
model.zero_grad()
del pruner


save_path = os.path.join(f"./pruned_model", p_string)
os.makedirs(save_path, exist_ok=True)

pipeline.save_pretrained(save_path)


os.makedirs(save_path, exist_ok=True)
torch.save(model, os.path.join(save_path, "unet_pruned.pth"))

# Sampling images from the pruned model
pipeline = DDIMPipeline(
    unet = model,
    scheduler = DDIMScheduler.from_pretrained(save_path, subfolder="scheduler")
)
with torch.no_grad():
    generator = torch.Generator(device=pipeline.device).manual_seed(25)
    pipeline.to("cpu")
    images = pipeline(num_inference_steps=1000, batch_size=1, generator=generator, output_type="numpy").images
    os.makedirs(os.path.join(save_path, 'vis'), exist_ok=True)
    torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(save_path))