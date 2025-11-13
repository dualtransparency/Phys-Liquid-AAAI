from typing import Tuple

import numpy as np
import torch
import json
from omegaconf import OmegaConf
from PIL import Image
from model import CRM
from pipelines import TwoStagePipeline
from inference import generate3d
from config import Config


class ModelManager:
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = None
        self.model = None

    def load_models(self):
        specs = json.load(open(self.config.specs_path))
        self.model = CRM(specs).to(self.config.device)
        self.model.load_state_dict(
            torch.load(self.config.crm_path, map_location=self.config.device),
            strict=False
        )

        stage1_config = OmegaConf.load(self.config.stage1_config).config
        stage2_config = OmegaConf.load(self.config.stage2_config).config

        self.pipeline = TwoStagePipeline(
            stage1_config.models,
            stage2_config.models,
            stage1_config.sampler,
            stage2_config.sampler,
            device=self.config.device,
            dtype=torch.float16
        )

    def generate_3d(
            self,
            input_image: Image.Image,
            seed: int,
            scale: float,
            step: int
    ) -> Tuple[Image.Image, Image.Image, str, str]:
        self.pipeline.set_seed(seed)
        rt_dict = self.pipeline(input_image, scale=scale, step=step)

        stage1_imgs = rt_dict["stage1_images"]
        stage2_imgs = rt_dict["stage2_images"]

        np_imgs = np.concatenate(stage1_imgs, axis=1)
        np_xyzs = np.concatenate(stage2_imgs, axis=1)

        return generate3d(
            self.model,
            np_imgs,
            np_xyzs,
            self.config.device
        )