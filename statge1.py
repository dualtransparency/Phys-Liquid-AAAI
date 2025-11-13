import torch
from omegaconf import OmegaConf
from PIL import Image
import os
import numpy as np
from imagedream.ldm.util import instantiate_from_config, get_obj_from_str
from run import preprocess_image

# Configuration parameters
stage1_model_config = OmegaConf.load('imagedream/configs/sd_v2_base_ipmv_zero_SNR.yaml')
stage1_model_config.resume = '/mnt/ssd/fyz/CRM/pixel-diffusion.pth'
stage1_model_config.resume_unet = '/home/fyz/CRM/unet-1000016'

# Load stage1 model
stage1_model = instantiate_from_config(stage1_model_config.model)
stage1_model.load_state_dict(torch.load(stage1_model_config.resume, map_location="cpu"), strict=False)
unet_state_dict = torch.load(stage1_model_config.resume_unet, map_location="cpu")
stage1_model.model.load_state_dict(unet_state_dict, strict=False)
# 加载新的模型参数文件
# stage1_model.load_state_dict(torch.load('stage1_finetuned_unet.pth', map_location='cpu'), strict=False)

device = "cuda"
dtype = torch.float16
stage1_model = stage1_model.to(device).to(dtype)
stage1_model.device = device

# Sampler configuration
stage1_sampler_config = {
    "target": "libs.sample.ImageDreamDiffusion",
    "params": {
        "mode": "pixel",
        "num_frames": 7,
        "camera_views": [1, 2, 3, 4, 5, 0, 0],
        "ref_position": 6,
        "random_background": False,
        "offset_noise": True,
        "resize_rate": 1.0
    }
}

# Initialize sampler
stage1_sampler = get_obj_from_str(stage1_sampler_config["target"])(
    stage1_model, device=device, dtype=dtype, **stage1_sampler_config["params"]
)

# Prepare input image
input_image_path = '/home/fyz/CRM/realtestmask.png'  # Replace with actual path
img = Image.open(input_image_path)
img = preprocess_image(img, background_choice="Auto Remove background", foreground_ratio=1.0, background_color=(127, 127, 127))

# Run stage1 inference
uc = stage1_sampler.model.get_learned_conditioning(["uniform low no texture ugly, boring, bad anatomy, blurry, pixelated, obscure, unnatural colors, poor lighting, dull, and unclear."]).to(device)
stage1_images = stage1_sampler.i2i(
    stage1_sampler.model,
    stage1_sampler.size,
    "3D assets",
    uc=uc,
    sampler=stage1_sampler.sampler,
    ip=img,
    step=50,
    scale=5,
    batch_size=stage1_sampler.batch_size,
    ddim_eta=0.0,
    dtype=dtype,
    device=device,
    camera=stage1_sampler.camera,
    num_frames=stage1_sampler.num_frames,
    pixel_control=(stage1_sampler.mode == "pixel"),
    transform=stage1_sampler.image_transform,
    offset_noise=stage1_sampler.offset_noise,
)

# Remove reference image
stage1_images.pop(stage1_sampler.ref_position)

# Save output images
output_dir = '/home/fyz/CRM/out'  # Replace with actual output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i, img_array in enumerate(stage1_images):
    # Handle invalid values
    img_array = np.nan_to_num(img_array, nan=0)
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    # Convert to PIL Image and save
    img = Image.fromarray(img_array)
    img.save(f'{output_dir}/output_stage1_{i}.png')