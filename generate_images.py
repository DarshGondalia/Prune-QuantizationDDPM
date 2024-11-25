import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torchvision
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
from tqdm import tqdm

# Define the path to the pruned models
pruned_models_dir = "./pruned_model"
# models = [os.path.join(pruned_models_dir, d) for d in os.listdir(pruned_models_dir) if os.path.isdir(os.path.join(pruned_models_dir, d))]
models = ["./pruned_model/0_3/"]
# Number of images to generate per model
num_images = 100
batch_size = 20  # Adjust based on your GPU memory

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_path in models:
    print(f"Processing model at: {model_path}")
    
    # Load the pruned UNet model
    # unet_model_path = os.path.join(model_path, "unet_pruned.pth")
    unet_model_path = os.path.join(model_path, "unet_finetuned.pth")
    if not os.path.exists(unet_model_path):
        print(f"UNet model not found at {unet_model_path}, skipping...")
        continue
    
    # # Load the pruned model
    unet = torch.load(unet_model_path, map_location=device)
    print(unet)
    unet.eval()
    unet.to(device)
    #Post Finetune
    # unet = UNet2DModel(...)  # Instantiate model
    # model.load_state_dict(torch.load(unet_model_path, map_location=device))
    # unet.eval()
    # unet.to(device)
    # Load the scheduler
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    
    # Set up the pipeline
    pipeline = DDIMPipeline(
        unet=unet,
        scheduler=scheduler
    )
    pipeline.to(device)
    
    # Directory to save generated images
    save_dir = os.path.join(model_path, 'generated_images')
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate images
    num_batches = num_images // batch_size
    image_counter = 0
    
    with torch.no_grad():
        for batch_num in tqdm(range(num_batches), desc=f"Generating images for model {model_path}"):
            generator = torch.Generator(device=device).manual_seed(batch_num)  # For reproducibility
            images = pipeline(
                num_inference_steps=700,
                batch_size=batch_size,
                generator=generator,
                output_type="numpy"
            ).images  # Returns a list of numpy arrays
            
            # Convert images to tensors and save
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)  # (batch_size, C, H, W)
            for i in range(images_tensor.size(0)):
                image = images_tensor[i]
                image_save_path = os.path.join(save_dir, f"generated_{image_counter:05d}.png")
                torchvision.utils.save_image(
                    image,
                    image_save_path,
                    normalize=True,
                    value_range=(-1, 1)
                )
                image_counter += 1

    print(f"Finished generating images for model at {model_path}")