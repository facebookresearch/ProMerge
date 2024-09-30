## [ECCV'24] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation

ProMerge is a fundamental improvement over prior leading approaches to unsupervised instance segmentation and object detection. For the challenging SA1B benchmark, we observe a 41.8% improvement in AP and 20.6% improvement in AR compared to the CutLER SOTA. We observe increases in AR and AP across six benchmarks. 

Official PyTorch implementation for ProMerge (ECCV'24). Details can be found in the paper.
[[`paper`](#)]
[[`project page`](https://www.robots.ox.ac.uk/~vgg/research/promerge/)]

![Alt Text](assets/overview.png)

### Features
- Prior leading unsupervised methods rely on repeatedly solving graph partitioning over the global context of the image in feature space. These methods miss smaller objects that have local context.
- We propose ProMerge, which generates a large number of masks per image by obviating the resolution of the generalized eigevalue problem. ProMerge lifts recall on diverse datasets while reducing pseudolabel generation time. 
- We use the [CutLER](https://github.com/facebookresearch/CutLER?tab=readme-ov-file) training recipe for a single round and show that training a MaskRCNN detector on the high-quality ProMerge pseudo-labels results in SOTA performance on six diverse benchmarks. 

### Demo
To be updated.

### Inference
Please download datasets and their annotation files:
- [COCO2017](http://images.cocodataset.org/zips/val2017.zip) [[`annotation file`](http://dl.fbaipublicfiles.com/cutler/coco/coco_cls_agnostic_instances_val2017.json)]
- [COCO-20K](https://cocodataset.org/#download) [[`annotation file`](http://dl.fbaipublicfiles.com/cutler/coco/coco20k_trainval_gt.json)]
- [LVIS](http://images.cocodataset.org/zips/val2017.zip) [[`annotation file`](http://dl.fbaipublicfiles.com/cutler/coco/lvis1.0_cocofied_val_cls_agnostic.json)]
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_step.php) [[`annotation file`](https://dl.fbaipublicfiles.com/cutler/kitti/trainval_cls_agnostic.json)]

<!-- - [Objects365](https://www.objects365.org/download.html) [[`annotation file`](#)]
- [SA-1B](https://scontent-lhr8-1.xx.fbcdn.net/m1/v/t6/An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0.tar?ccb=10-5&oh=00_AYA9kGsV-zzziVDpf8ErkuQzkQ4GW2nYfw8RsFN9aosqhg&oe=66F7EB7E&_nc_sid=0fdd51) [[`annotation file`](#)] -->

<!-- Note that KITTI and Objects365 require you to sign up to download the data. -->

### Result files
To be uploaded.
<!-- We provide predictions for each dataset as follows.
#### ProMerge
| dataset  | AP50 | AP | AR | output file |
|----------|------|----|----|-------------|
| COCO2017 |      |    |    |             |
| COCO-20K |      |    |    |             |
| LVIS     |      |    |    |             |
| KITTI    |      |    |    |             |
| SA-1B    |      |    |    |             |

#### ProMerge+
| dataset  | AP50 | AP | AR | output file |
|----------|------|----|----|-------------|
| COCO2017 |      |    |    |             |
| COCO-20K |      |    |    |             |
| LVIS     |      |    |    |             |
| KITTI    |      |    |    |             |
| SA-1B    |      |    |    |             | -->

### License 
The majority of ProMerge, Detectron2 and DINO are licensed under the CC-BY-NC license. However portions of the project are available under separate license terms. CRF is licensed under the MIT license. If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0.

### Citation
```
@inproceedings{li2024promerge,
  title = {ProMerge: Prompt and Merge for Unsupervised Instance Segmentation},
  author = {Li, Dylan and Shin, Gyungin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}
```

### Questions
If you have any questions about our code/implementation, please contact us at gyungin [at] robots [dot] ox [dot] ac [dot] uk.
