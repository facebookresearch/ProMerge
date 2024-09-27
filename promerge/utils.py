# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import List
import torch
import numpy as np
import PIL.Image as Image

from detectron2.utils.colormap import random_color
import torch.nn.functional as F
import matplotlib.pyplot as plt


def get_mask_affinities(mask1, mask2, features):
    affinities = []
    true_indices_m1 = torch.nonzero(mask1.flatten(), as_tuple=False).view(-1)
    true_indices_m2 = torch.nonzero(mask2.flatten(), as_tuple=False).view(-1)
    shuffled_indices_m1 = torch.randperm(true_indices_m1.size(0))
    shuffled_indices_m2 = torch.randperm(true_indices_m2.size(0))
    random_points_indices_m1 = true_indices_m1[shuffled_indices_m1[:10]]
    random_points_indices_m2 = true_indices_m2[shuffled_indices_m2[:10]]
    for r1 in random_points_indices_m1:
        for r2 in random_points_indices_m2:
            r1_row = r1 // mask1.shape[0]
            r1_col = r1 % mask1.shape[0]
            r2_row = r2 // mask2.shape[0]
            r2_col = r2 % mask2.shape[0]
            affinity = features[:, r1_row, r1_col] @ features[:, r2_row, r2_col]
            affinities.append(affinity)
            
    return affinities

def check_num_fg_sides(mask):
    width, height = mask.shape[0], mask.shape[1]
    extension = 1
    top, right, bottom, left = torch.sum(mask[:extension, :]) / (extension * width), torch.sum(mask[:, -1 * extension:]) / (
        extension * height), torch.sum(mask[-1 * extension:, :]) / (extension * width), torch.sum(mask[:, :extension]) / (extension * height)
    nc = torch.round(top) + torch.round(right) + \
        torch.round(bottom) + torch.round(left)
    return nc

# plotting all remaining foreground masks
def plot_masks(I_int: Image, masks: List[torch.tensor], img_out_dir: str, img_path: str, ext: str):
    for fg_mask in masks:
        pseudo_mask = Image.fromarray(np.uint8(np.uint8(fg_mask.detach().cpu() >= 1) * 255))
        pseudo_mask = np.asarray(pseudo_mask.resize((I_int.width, I_int.height)))
        mask_color = random_color(rgb=False, maximum=255)
        fg = pseudo_mask > 0.5
        rgb = np.copy(I_int)
        rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color)
                * 0.7).astype(np.uint8)
        I_int = Image.fromarray(rgb)
    I_int.save(
            f"{img_out_dir}/{img_path.split('/')[-1].split('.')[0]}_{ext}.jpg")

def dino_similarity(mask1: torch.tensor, mask2: torch.tensor, dino_features: torch.tensor, dino_merge_tau: float) -> bool:
    mask1_dino_ft = dino_features[:, mask1.bool()]
    mask2_dino_ft = dino_features[:, mask2.bool()]
    
    mask1_avg_ft = torch.mean(mask1_dino_ft, dim=-1)
    mask2_avg_ft = torch.mean(mask2_dino_ft, dim=-1)
    assert mask1_avg_ft.shape == torch.Size([768])
    assert mask2_avg_ft.shape == torch.Size([768])
    
    product = mask1_avg_ft @ mask2_avg_ft
    
    if product < dino_merge_tau: 
        return False
    return True

def IoU(mask1, mask2):
    mask1, mask2 = (mask1>0.5).to(torch.bool), (mask2>0.5).to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()

def IoU_np(mask1, mask2):
    mask1, mask2 = np.array(mask1 > 0.5, dtype=bool), np.array(mask2 > 0.5, dtype=bool)
    intersection = np.sum(mask1 & mask2, axis=(-1, -2)).squeeze()
    union = np.sum(mask1 + mask2, axis=(-1, -2)).squeeze()
    return (intersection / union).mean().item()

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h