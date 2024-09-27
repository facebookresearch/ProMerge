# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python3 promerge.py --out_dir annotations_out \
--vit_arch base --dataset_path ../../unsupervised_video/datasets/val2017/ \
--fixed_size 480 \
--stride 4 \
--bipartition_type threshold \
--bipartition_tau 0.2 \
--vit_feat k \
--foreground_include_tau 0.1 \
--intersection_merge_tau 0.5 \
--dino_merge_tau 0.1 \
--cc_maskarea_tau 0.5 \
--merge_ioa_tau 0.1 \
--iou_filter_tau 0.8 \
--suffix promerge_dir
