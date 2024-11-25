from pytorch_fid import fid_score

real_image_dir = '/Users/darshgondalia/Desktop/CS 682/Final Project/real_images_10'
image_save_dir = '/Users/darshgondalia/Desktop/CS 682/Final Project/pruned_model/0_3/generated_images'

fid_value = fid_score.calculate_fid_given_paths(
    [real_image_dir, image_save_dir],
    batch_size=10,
    device="cpu",
    dims=2048,
    num_workers=0 
)

print(f"FID score after fine-tuning: {fid_value}")