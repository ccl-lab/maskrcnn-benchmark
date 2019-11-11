# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import json
import pickle
from PIL import Image
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import pycocotools.mask as mask_util
from torchvision.transforms import functional as F
from torchvision import transforms as T
import pdb
import cv2
import os.path as osp

from .coco import COCODataset

# Dataloader for shortest path (SP) baseline
# TODO: Could be merged into EVRDataset with a new condition
class HOMEDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, ann_file, root, seq_len=10, seq_step=1, transforms=None, target_transform=None):
        from pycocotools.coco import COCO
        root = osp.join(osp.abspath(osp.join(osp.dirname(__file__), '..', '..', "..")), root)
        self.root = root
        ann_file = osp.join(osp.abspath(osp.join(osp.dirname(__file__), '..', '..', "..")), ann_file)

        super(HOMEDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        self.categories = {
            1: "bison",
            2: "alligator",
            3: "drop",
            4: "kettle",
            5: "koala",
            6: "lemon",
            7: "mango",
            8: "moose",
            9: "pot",
            10: "seal"
        }

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        self.annos = json.load(open(ann_file))
        print()


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        img, anno = super(HOMEDataset, self).__getitem__(idx)
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx


    def get_img_info(self, index):
        return {"height": self.annos['images'][index]['height'],
                "width": self.annos['images'][index]['width'],
                # "vis_ratio": self.annos['annotations'][index]['vis_ratio']
                }
                
    def get_seq_step(self):
        return self.seq_step

    def get_obj_category(self):
        return self.annos['categories']
