import torch

# Load the original stage1 model's state dict
stage1_state_dict = torch.load('/mnt/ssd/fyz/CRM/pixel-diffusion.pth', map_location='cpu')

# Load the fine-tuned UNet's state dict
unet_finetuned_state_dict = torch.load('/home/fyz/CRM/unet-1000016', map_location='cpu')

# Define the prefix for the UNet in the original model
unet_prefix = 'model.diffusion_model.'

# Create a new state dict based on the original stage1 model
new_state_dict = stage1_state_dict.copy()

# Iterate through the fine-tuned UNet's state dict
for key in unet_finetuned_state_dict:
    # Prepend the prefix
    new_key = unet_prefix + key
    # Check if this new_key exists in the original state dict
    if new_key in new_state_dict:
        # Update the value
        new_state_dict[new_key] = unet_finetuned_state_dict[key]
    else:
        # Print a warning if the key doesn't exist
        print(f"Warning: Key {new_key} not found in the original model.")

# Now, save the new state dict
torch.save(new_state_dict, 'stage1_finetuned_unet.pth')

# In the main script, load the new model
# Example:
# from imagedream.ldm.util import instantiate_from_config
# stage1_model_config = OmegaConf.load('imagedream/configs/sd_v2_base_ipmv_zero_SNR.yaml')
# stage1_model = instantiate_from_config(stage1_model_config.model)
# stage1_model.load_state_dict(torch.load('stage1_finetuned_unet.pth', map_location='cpu'), strict=False)
# device = "cuda"
# dtype = torch.float16
# stage1_model = stage1_model.to(device).to(dtype)