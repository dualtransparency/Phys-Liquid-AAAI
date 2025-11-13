import os
from omegaconf import OmegaConf

class Config:
    def __init__(self):
        self.device = "cuda"
        self.stage1_config = OmegaConf.load("../configs/nf7_v3_SNR_rd_size_stroke.yaml").config
        self.stage2_config = OmegaConf.load("../configs/stage2-v2-snr.yaml").config
        self.crm_path = "/mnt/ssd/fyz/CRM/CRM.pth"
        self.specs_path = "../configs/specs_objaverse_total.json"
        self.xyz_path = "/mnt/ssd/fyz/CRM/ccm-diffusion.pth"
        self.pixel_path = "/mnt/ssd/fyz/CRM/pixel-diffusion.pth"
        self.examples_dir = "liquid_examples"
        self.mechanism_dir = "mechanism_examples"