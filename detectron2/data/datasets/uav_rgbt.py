# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_uav_rgbt_instances", "register_uav_rgbt"]


# fmt: off
CLASS_NAMES = (
    "person", "dog0", "bicycle", "sports ball", "car",  "boat", "motorcycle", "truck", "baby carriage",  "bus",  
)
CLASS_NAMES = (
    "person", "car",  "motorcycle" 
)
# fmt: on


def load_uav_rgbt_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    # if split == 'test':
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    # else:
        # annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Gen_annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        rgb_file = os.path.join(dirname, "rgb", fileid + ".png")
        thr_file = os.path.join(dirname, "trm", fileid + ".png")


        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": [rgb_file, thr_file],
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls == 'truck' or cls ==  "bus":
                cls = "car"
            if cls not in CLASS_NAMES:
                continue
            # if CLASS_NAMES.index(cls)  <= 5 and 'val' in split:
            #     continue
            # split data 
            
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_uav_rgbt(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_uav_rgbt_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
