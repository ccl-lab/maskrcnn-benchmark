import cv2

import torch

import sys
import argparse
import json
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image

from util import load_json, save_json, process_boxes

CATEGORIES = ['helmet', 'house', 'blue car', 'rose', 'elephant',
              'snowman', 'rabbit', 'spongebob', 'turtle', 'gavel',
              'ladybug', 'mantis', 'green car', 'saw', 'puppet',
              'phone', 'r. cube', 'rake', 'truck', 'white car',
              'rattle', 'p. cube', 'bed', 't. cube']

parser = argparse.ArgumentParser(description='Add boxes to image frames')
parser.add_argument("--image_dir", type=str) # path to directory with original images
parser.add_argument("--boxes", type=str) # path to bbox.json file output by model
parser.add_argument("--data", type=str) # path to annotation dataset json file
parser.add_argument("--out_dir", type=str) # path to output dir

args = parser.parse_args()

img_dir = args.image_dir
boxes = load_json(args.boxes)
boxes = process_boxes(boxes, min_thresh=0.75)
data = load_json(args.data)
outdir = args.out_dir

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )

    return image


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def map_boxes_to_imgs(boxes, data):
    results = {}
    for img in data['images']:
        fname = img['file_name']
        id = img['id']
        results[fname] = []
        for b in boxes:
            if b['image_id'] == id:
                results[fname].append(b)
    return results

def json2boxlist(imgs):
    results = {}
    for img, boxs in imgs.items():
        i = Image.open(os.path.join(img_dir, img))
        boxes = [obj["bbox"] for obj in boxs]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, i.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in boxs]
        # classes = [c for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        scores = [obj['score'] for obj in boxs]
        scores = torch.tensor(scores)
        target.add_field("scores", scores)

        results[img] = target

    return results

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    # print(predictions)
    labels = predictions.get_field("labels").tolist()
    labels = [CATEGORIES[i-1] for i in labels]
    print(labels)
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, label, score in zip(boxes, labels, scores):
        # if label == "pot":
        #     print()
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2
        )

    return image


img_boxes = map_boxes_to_imgs(boxes, data)
img_boxes = json2boxlist(img_boxes)

results = []

if not os.path.isdir(outdir):
    os.mkdir(outdir)

for img, boxes in img_boxes.items():
    if len(boxes) > 0:
        print(img)
        i = cv2.imread(os.path.join(img_dir, img))
        i = overlay_boxes(i, boxes)
        i = overlay_class_names(i, boxes)
        cv2.imwrite(os.path.join(outdir, img), i)
    # results.append((img, i))



# for i in results:
#     cv2.imwrite(os.path.join(outdir, i[0]), i[1])