# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
from glob import glob
import os
from pathlib import Path

from crf import densecrf
import numpy as np
import PIL
import PIL.Image as Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import dino
from utils import plot_masks, resize_pil, IoU, check_num_fg_sides, dino_similarity


class ProMerge:
    # Image transformation applied to all images for DINO transformer
    ToTensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def __init__(self, args) -> None:
        self.args = args

        self.image_list = sorted(glob(f"{args.image_dir}/*"))

        self.patch_size = 8
        self.feat_h = int(args.fixed_size / self.patch_size)
        self.feat_w = int(args.fixed_size / self.patch_size)

        # visualization options
        self._plot_figures = bool(self.args.img_out_dir is not None)
        self._plot_final_figures = bool(self.args.final_figures_dir is not None)

    def calculate_affinity_matrix(self, dino_features: torch.tensor, seed: torch.tensor, eps=1e-5):
        """Inner product between seed feature and every feature in DINO"""
        seed = seed.unsqueeze(-1).unsqueeze(-1)
        dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + eps)
        seed = seed / (torch.linalg.norm(seed, axis=0) + eps)
        affinity = (dino_features * seed).sum(dim=0)
        assert affinity.shape == dino_features.shape[1:]
        return affinity

    def generate_dino_affinities(self, dino_features, feat_w, feat_h):
        masked_foreground_affinities = []
        masked_background_affinities = []
        for i in range(0, feat_w, self.args.stride):
            for j in range(0, feat_h, self.args.stride):
                seed_feature = dino_features[:, i, j]
                normalized_affinity = self.calculate_affinity_matrix(dino_features, seed_feature)

                masked_affinity = (normalized_affinity > self.args.bipartition_tau).float()

                fg_sides = check_num_fg_sides(masked_affinity)
                if fg_sides > 1:
                    # background candidate
                    masked_background_affinities.append(masked_affinity)
                    continue

                # mask splitting
                objects_ccs, n_ccs = ndimage.label(masked_affinity.cpu().numpy())
                object_sizes = []
                for obj_idx in range(1, n_ccs+1):
                    object_sizes.append(
                        (obj_idx, np.sum(objects_ccs[objects_ccs == obj_idx]) / obj_idx))

                # find the object corresponding to the center, then find its size
                base_object_size = object_sizes[objects_ccs[i, j] - 1][1]
                object_sizes.sort(key=lambda x: x[1], reverse=True)

                tmp = []
                for object in object_sizes:
                    idx, size = object[0], object[1]
                    if size < self.args.cc_maskarea_tau * base_object_size:
                        break
                    masked_cc = np.zeros_like(objects_ccs)
                    masked_cc[objects_ccs == idx] = 1
                    tmp.append(masked_cc)
                    masked_cc_torch = torch.from_numpy(masked_cc).cuda()
                    masked_foreground_affinities.append((masked_cc_torch, torch.sum(masked_cc_torch)))

        return masked_foreground_affinities, masked_background_affinities

    def cascade_filter(self, masked_foreground_affinities, normalized_dino_features, background):
        # store the mask with the most corners, but least masked area
        cascade_filtered_masks = []

        # sort the foreground masks by their size.
        masked_foreground_affinities.sort(key=lambda x: x[1])
        combined_foreground = torch.zeros_like(background)

        background_mean_ft = torch.mean(normalized_dino_features[:, background.bool()], dim=-1)
        assert background_mean_ft.shape == torch.Size([768]), f"{background_mean_ft.shape}"

        # process the foreground masks, possibly use mask splitting
        for mask, _ in masked_foreground_affinities:
            new_mask_points = torch.clone(mask)
            new_mask_points[combined_foreground > 0] = 0

            mask_mean_ft = torch.mean(
                normalized_dino_features[:, mask.bool()], dim=-1)
            assert mask_mean_ft.shape == torch.Size([768])

            if (torch.sum(new_mask_points * background) / torch.sum(new_mask_points)) < self.args.iou_filter_tau \
                    and mask_mean_ft @ background_mean_ft < self.args.foreground_include_tau:
                combined_foreground += mask
                cascade_filtered_masks.append((mask > 0).to(torch.uint8))

        return cascade_filtered_masks

    def resize_image(self, image: Image.Image):
        resized_image = image.resize((int(self.args.fixed_size), int(self.args.fixed_size)), PIL.Image.LANCZOS)
        resized_image, _, _, feat_w, feat_h = resize_pil(resized_image, self.patch_size)

        return resized_image, feat_w, feat_h

    def __call__(self, backbone_dino):
        for image_path in tqdm(self.image_list):
            image_name = os.path.basename(image_path).split(".")[0]
            image: Image.Image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.width, image.height

            # resize the image to fixed size
            resized_image, feat_w, feat_h = self.resize_image(image)

            # convert the image to tensor
            image_tensor = self.ToTensor(resized_image).unsqueeze(0).cuda()

            dino_features_raw = backbone_dino(image_tensor)[0]
            dino_features = dino_features_raw.reshape((dino_features_raw.shape[0], feat_w, feat_h)).detach()

            masked_foreground_affinities, masked_background_affinities = self.generate_dino_affinities(
                dino_features, feat_w, feat_h
            )

            if len(masked_foreground_affinities) == 0:
                print(f"No foreground masks for img: {image_path}")
                continue

            normalized_dino_features = dino_features / (torch.linalg.norm(dino_features, axis=0) + 1e-5)

            # plotting the aggregated background information
            if len(masked_background_affinities):
                # pixel-wise voting for background
                masked_background_affinities_agg = torch.stack(masked_background_affinities)
                background_summed = torch.mean(masked_background_affinities_agg, axis=0)
                background = (background_summed > 0.5).to(torch.int)

            else:
                continue  # no annotations if no background is detected

            if torch.sum(background) / (background.shape[0] * background.shape[1]) < 0.1:
                print(f"Skipping masks for {image_path}, background insufficient size")
                continue

            ############################# foreground filtering ##############################
            cascade_filtered_masks = self.cascade_filter(
                masked_foreground_affinities, normalized_dino_features, background
            )
            #################################################################################

            # contains a set of (masks)
            clustered_masks = set()
            cascade_filtered_masks.sort(key=lambda x: torch.sum(x), reverse=True)

            # merging happens here
            for mask in cascade_filtered_masks:
                if torch.sum(mask) == 0:
                    continue
                if len(clustered_masks) == 0:
                    clustered_masks.add(mask)
                    continue
                masks_to_combine = []
                for cluster_mask in clustered_masks:
                    # skip over empty masks
                    intersection = cluster_mask & mask
                    intersect_area = torch.sum(intersection)
                    if intersect_area == 0:
                        continue
                    # if section IOU > threshold, combine
                    intersection_over_mask_area = intersect_area / torch.sum(mask)
                    if intersection_over_mask_area > self.args.intersection_merge_tau:
                        masks_to_combine.append(cluster_mask)
                    elif intersection_over_mask_area > self.args.merge_ioa_tau and dino_similarity(
                        mask,
                        cluster_mask,
                        normalized_dino_features,
                        self.args.dino_merge_tau
                    ):
                        masks_to_combine.append(cluster_mask)

                # if not added to an existing cluster of masks, add to a new cluster of masks. Else, combine
                # all masks that match the certain criteria of having correlation in depth or overlap in mask
                # area.
                if len(masks_to_combine) == 0:
                    clustered_masks.add(mask)
                else:
                    combined_mask = torch.zeros_like(mask)
                    for mask_to_combine in masks_to_combine:
                        clustered_masks.remove(mask_to_combine)
                        combined_mask += mask_to_combine
                    combined_mask += mask
                    combined_mask = (combined_mask > 0).to(torch.uint8)
                    clustered_masks.add(combined_mask)

            clustered_masks_upsampled = set()

            # postprocess and refine final masks
            for clustered_mask in clustered_masks:
                clustered_mask_upsampled = F.interpolate(
                    clustered_mask.unsqueeze(0).unsqueeze(0),
                    size=(self.args.fixed_size, self.args.fixed_size),
                    mode='nearest'
                ).squeeze()
                clustered_masks_upsampled.add(clustered_mask_upsampled.cuda())

            clustered_masks_upsampled_list = list(clustered_masks_upsampled)
            
            if len(clustered_masks_upsampled_list) == 0:
                continue

            # plot all masks on the image as separate objects
            I_new = resized_image
            masks_final = []
            for pseudo_mask in clustered_masks_upsampled_list:
                pseudo_mask = np.float32(pseudo_mask.cpu() >= 1)
                assert pseudo_mask.shape == (self.args.fixed_size, self.args.fixed_size)
                pseudo_mask_crf = densecrf(np.array(I_new), pseudo_mask)
                pseudo_mask_crf = ndimage.binary_fill_holes(pseudo_mask_crf >= 0.5)
                mask1 = torch.from_numpy(pseudo_mask_crf).cuda()
                mask2 = torch.from_numpy(pseudo_mask).cuda()
                if IoU(mask1, mask2) < 0.5:
                    continue
                pseudo_mask_crf = np.uint8(pseudo_mask_crf * 255)
                pseudo_mask_crf = Image.fromarray(pseudo_mask_crf)
                pseudo_mask_crf = np.asarray(pseudo_mask_crf.resize((image_width, image_height)))

                if self._plot_figures or self._plot_final_figures:
                    masks_final.append(torch.from_numpy(pseudo_mask_crf.copy()))

            if self._plot_figures or self._plot_final_figures:
                dir_final_images = Path(
                    self.args.final_figures_dir
                ) if self._plot_final_figures else Path(self.args.img_out_dir)

                plot_masks(
                    image.resize((image_width, image_height), PIL.Image.LANCZOS),
                    masks_final,
                    dir_final_images,
                    image_path,
                    "final"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ProMerge script')

    # default arguments
    parser.add_argument('--vit_arch', type=str, default='base',
                        choices=['base', 'small', 'base_v2'], help='which architecture')
    parser.add_argument('--suffix', type=str, default='', help='suffix of annotations output file')
    parser.add_argument('--vit_feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--dataset_path', type=str, default="imagenet/train/", help='path to the dataset')
    parser.add_argument('--fixed_size', type=int, default=480,
                        help='rescale the input images to a fixed size')
    parser.add_argument('--stride', type=int, default=4,
                        help='stride over dino features for seed points. If dino features are 60x60, stride is 6 (100 total seeds), then there will be 100 mask proposals.')
    parser.add_argument('--bipartition_type', type=str,
                        choices=['tokencut', 'threshold', 'threshold_fp_base', 'no_bipartitioning'],
                        help='method used to bipartition each correlation matrix')
    parser.add_argument('--bipartition_tau', type=float, default=0.2,
                        help='Used either as raw threshold, or as tokencut threshold, depending on selected bipartition_type')
    parser.add_argument('--bipartition_iterations', type=int, default=1,
                        help='Only used for threshold_fp_base, used to determine bipartition iterations')
    parser.add_argument('--iou_filter_tau', type=float, default=0.8,
                        help='filter for mask intersection during fg filtering process')
    parser.add_argument('--merge_ioa_tau', type=float, default=0.1,
                        help='intersection over area for dino merge supplement')
    parser.add_argument('--foreground_include_tau', type=float, default=0.1,
                        help='threshold to consider separating including foreground mask in merge process')
    parser.add_argument('--intersection_merge_tau', type=float, default=0.5,
                        help='threshold to consider separating including foreground mask in merge process')
    parser.add_argument(
        '--dino_merge_tau', type=float, default=0.1,
        help='dino affinity threshold to consider merging foreground masks in depth merge'
    )
    parser.add_argument(
        '--cc_maskarea_tau', type=float, default=0.5,
        help='if the cc has area less than cc_maskarea_tau * (# of pixels in CC containing seed pt) # of pixels, do not consider cc'
    )
    parser.add_argument(
        "--crf_iou_tau", type=float, default=0.5,
    )

    # annotation out parameters
    parser.add_argument(
        '--full_annotations', action='store_true', default=False,
        help='Verbose dumping of image metadata and annotations.'
    )

    # project directory
    parser.add_argument(
        "--project_dir", type=str, default="/home/cs-shin1/ProMerge",
        help="path to the ProMerge directory"
    )

    # image directory
    parser.add_argument(
        "--image_dir", type=str, default="assets/example_images",
        help="path to the image directory"
    )

    # visualisations
    parser.add_argument(
        '--img_out_dir', type=str, default="assets/example_outputs",
        help='output intermediary images for fast adhoc visual eval'
    )
    parser.add_argument(
        '--final_figures_dir', type=str, default=None,
        help='output final images for fast adhoc visual eval'
    )

    # path to monocular depth model params
    args = parser.parse_args()

    assert os.path.exists(args.project_dir), f"Project directory {args.project_dir} does not exist"
    assert (args.img_out_dir is None) or (args.final_figures_dir is None)

    # set up absolute paths for directories
    args.image_dir = f"{args.project_dir}/{args.image_dir}"
    args.img_out_dir = f"{args.project_dir}/{args.img_out_dir}"
    print(f"Making predictions for images in directory: {args.image_dir}")

    if args.img_out_dir is not None:
        os.makedirs(args.img_out_dir, exist_ok=True)
    if args.final_figures_dir is not None:
        os.makedirs(args.final_figures_dir, exist_ok=True)

    if args.vit_arch == 'base':
        url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small':
        url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    print(f'Load {args.vit_arch} pre-trained feature...')
    backbone_dino = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, 8)

    backbone_dino.eval()
    backbone_dino.cuda()

    image_paths = []
    promerge = ProMerge(args)
    promerge(backbone_dino)
