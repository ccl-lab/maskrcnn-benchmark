from PIL import Image

import argparse
import json
import random
import os.path as osp
import os

def choose_random(boxes, num):
    chosen = {}

    random.shuffle(boxes)

    boxes = boxes[:num*50] if len(boxes) > num*50 else boxes

    for i in range(1, 13):
        obj_boxes = []
        for b in boxes:
            if b['category_id'] == i:
                obj_boxes.append(b)

        random.shuffle(obj_boxes)
        if len(obj_boxes) < num:
            chosen[i] = obj_boxes
        else:
            chosen[i] = obj_boxes[:num]

    return chosen


def splice_objects(boxes, img_dir, output):
    print()
    for i, boxes in boxes.items():
        out_dir = osp.join(output, str(i))
        if not osp.isdir(out_dir):
            os.mkdir(out_dir)
        for j, b in enumerate(boxes):
            img = Image.open(osp.join(img_dir, b['fname']))
            p = list(map(int, b['bbox']))
            cropped = img.crop((p[0], p[1], p[0]+p[2], p[1]+p[3]))
            cropped.save(osp.join(out_dir, "{}.jpg".format(j)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox")
    parser.add_argument("--img_dir")
    parser.add_argument("--output")
    parser.add_argument("--num_random", type=int)
    args = parser.parse_args()

    with open(args.bbox, "r") as input:
        boxes = json.load(input)

    chosen = choose_random(boxes, args.num_random)

    splice_objects(chosen, args.img_dir, args.output)

    print()