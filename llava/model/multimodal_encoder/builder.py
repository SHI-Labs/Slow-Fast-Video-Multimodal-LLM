# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from .clip_encoder import CLIPVisionTower, SiglipVisionTower
from copy import deepcopy
from .convnext_encoder import ConvNextVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if "clip" in vision_tower and vision_tower.startswith("openai"):
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)       
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
    elif "siglip" in vision_tower:
        vision_tower_cfg.input_image_size = 384
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)  
    
    elif vision_tower == "convnext-576":
        ## ConvNeXt
        convnext_args = deepcopy(vision_tower_cfg)
        convnext_args.freeze_vision = False
        convnext_args.input_image_size = 576
        convnext_vision_tower = "convnext_xxlarge.clip_laion2b_soup" # hardcode
        return ConvNextVisionTower(convnext_vision_tower, convnext_args)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')
