# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

def filter_ioa(masked_foreground_affinities, normalized_dino_features, background, args):
    combined_mask_components = []
    for mask, _ in masked_foreground_affinities:
        if torch.sum(mask * background) / torch.sum(mask) < args.foreground_include_tau:
            combined_mask_components.append((mask > 0).to(torch.uint8))
            
    return combined_mask_components


def filter_cascade_v2(masked_foreground_affinities, normalized_dino_features, background, args, use_cascade=True):
    max_masks = 5000
    mean_vecs = torch.zeros(max_masks, 768).cuda()  # Preallocate tensor
    
    # sort the foreground masks by their size. 
    masked_foreground_affinities.sort(key=lambda x: x[1])
    background_mean_ft = torch.mean(normalized_dino_features[:, background.bool()], dim=-1)
    assert background_mean_ft.shape == torch.Size([768])

    # filter the foreground masks
    mean_vecs_row = 0
    filtered_mask_components = []
    for mask, _ in masked_foreground_affinities:
        mask_mean_ft = torch.mean(normalized_dino_features[:, mask.bool()], dim=-1)
        assert mask_mean_ft.shape == torch.Size([768])
        # higher mean_vec_tau rate increases the number of masks allowed in mean_vec
        if mask_mean_ft @ background_mean_ft < args.foreground_include_tau:
            mean_vecs[mean_vecs_row, :] = mask_mean_ft
            filtered_mask_components.append((mask > 0).to(torch.uint8))
            if mean_vecs_row < max_masks - 1:
                mean_vecs_row += 1
            
    # move onto next image
    if mean_vecs_row == 0:
        return None

    # remove masks based on cascade algorithm. 
    if not use_cascade:
        combined_mask_components = filtered_mask_components
    else: 
        # at this point, we have a limited set of foreground masks, and their associated mean vectors
        combined_foreground = torch.zeros_like(background)
        combined_mask_components = []
        for mask in filtered_mask_components:
            # add mask because it does not overlap with the background
            new_mask_points = torch.clone(mask)
            new_mask_points[combined_foreground > 0] = 0
            if torch.sum(new_mask_points) == 0:
                continue
            
            new_mask_mean_ft = torch.mean(normalized_dino_features[:, new_mask_points.bool()], dim=(-1))
            assert new_mask_mean_ft.shape == torch.Size([768])
            
            max_affinity_fg_mask = mean_vecs[:mean_vecs_row] @ new_mask_mean_ft
                
            # filter by low overlap with background and low variance of depth within a mask
            if torch.max(max_affinity_fg_mask) > new_mask_mean_ft @ background_mean_ft:
                combined_foreground += mask
                combined_mask_components.append((mask > 0).to(torch.uint8))
                mean_vecs[mean_vecs_row, :] = new_mask_mean_ft
                if mean_vecs_row < max_masks - 1:
                    mean_vecs_row += 1
                    
    return combined_mask_components


def filter_cascade_v1(masked_foreground_affinities, normalized_dino_features, background, args):
    # store the mask with the most corners, but least masked area
    combined_mask_components = []

    # sort the foreground masks by their size. 
    masked_foreground_affinities.sort(key=lambda x: x[1])
    combined_foreground = torch.zeros_like(background)

    background_mean_ft = torch.mean(normalized_dino_features[:, background.bool()], dim=-1)
    assert background_mean_ft.shape == torch.Size([768])

    # process the foreground masks, possibly use mask splitting
    for mask, _ in masked_foreground_affinities:
        new_mask_points = torch.clone(mask)
        new_mask_points[combined_foreground > 0] = 0
        
        mask_mean_ft = torch.mean(normalized_dino_features[:, mask.bool()], dim=-1)
        assert mask_mean_ft.shape == torch.Size([768])
        
        if (torch.sum(new_mask_points * background) / torch.sum(new_mask_points)) < args.iou_filter_tau \
            and mask_mean_ft @ background_mean_ft < args.foreground_include_tau:
            combined_foreground += mask
            combined_mask_components.append((mask > 0).to(torch.uint8))
    
    return combined_mask_components

def filter_cascade_fgbg_similarity(masked_foreground_affinities, normalized_dino_features, background):
    # store the mask with the most corners, but least masked area
    combined_mask_components = []

    # sort the foreground masks by their size. 
    masked_foreground_affinities.sort(key=lambda x: x[1])
    combined_foreground = torch.zeros_like(background)

    background_mean_ft = torch.mean(normalized_dino_features[:, background.bool()], dim=-1)
    assert background_mean_ft.shape == torch.Size([768])
    
    fg_masks_fts = []
    for fg_mask in masked_foreground_affinities:
        fg_masks_fts.append(torch.mean(normalized_dino_features[:, fg_mask[0].bool()], dim=-1))

    combined_forground = torch.zeros_like(masked_foreground_affinities[0][0])
    combined_forground += masked_foreground_affinities[0][0]
    combined_mask_components.append((masked_foreground_affinities[0][0] > 0).to(torch.uint8))
    # process the foreground masks, possibly use mask splitting
    for i, (mask, _) in enumerate(masked_foreground_affinities[1:]):
        new_mask_points = torch.clone(mask)
        new_mask_points[combined_foreground > 0] = 0
        
        mask_mean_ft = torch.mean(normalized_dino_features[:, new_mask_points.bool()], dim=-1)
        assert mask_mean_ft.shape == torch.Size([768])
        
        for ft in fg_masks_fts:
            if mask_mean_ft @ background_mean_ft < mask_mean_ft @ ft:
                combined_foreground += mask
                combined_mask_components.append((mask > 0).to(torch.uint8))
    
    return combined_mask_components
