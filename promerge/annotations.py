# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import pycocotools.mask as mask_util
import numpy as np
import PIL.Image as Image


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def create_annotation_info(annotation_id, image_id, binary_mask,
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(
            binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask_util.encode(
        np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask_util.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask_util.toBbox(binary_mask_encoded)

    rle = mask_util.encode(
        np.array(binary_mask[..., None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1.0,
        "iscrowd": 0,
        "area": area.tolist(),
        "score": 1.0,  # added as a constant for the evaluation pipeline
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info
