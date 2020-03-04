import numpy as np
import sys
import json


def filter_top(boxes, thresh=None, min_thresh=0.0):
    result = {}
    print(thresh)
    print(min_thresh)

    for key, item in boxes.items():
        if key not in result:
            result[key] = {}
        for key_cat, item_cat in item.items():
            if len(item_cat) == 1:
                # TODO: add min_thresh filtering
                result[key][key_cat] = item_cat[0]
            else:
                # only keep the top detection
                if thresh is None:
                    sorted_items = sorted(item_cat,
                                          key=lambda x: x['score'],
                                          reverse=True)
                    if sorted_items[0]['score'] >= min_thresh:
                        print(sorted_items[0])
                        result[key][key_cat] = sorted_items[0]

                # keep all detections with confidence >= thresh
                else:
                    for i in item_cat:
                        if i['score'] >= thresh:
                            result[key][key_cat] = i
    return result


def filter_overlap(boxes, thresh=None):
    """
    boxes is a list of boxes in unchunked form.

    return a filtereed version of the boxes without overlap
    """

    for i, boxi in enumerate(boxes):
        for j, boxj in enumerate(boxes):
            if overlap(boxi, boxj, thresh):
                print()


def overlap(box1, box2, thresh=None):
    b1 = box1['bbox']
    b2 = box2['bbox']

    # print()


def chunk_by_image(boxes):
    """
    turn a flat list of boxes into a hierarchy of:

    image
        category
            [boxes]

    :param boxes: list of box detections
    :return: dictionary of boxes chunked by image/category
    """
    chunks = {}
    for b in boxes:

        if b['image_id'] not in chunks:
            chunks[b['image_id']] = {b['category_id']: [b]}
        elif b['category_id'] not in chunks[b['image_id']]:
            chunks[b['image_id']][b['category_id']] = [b]
        else:
            chunks[b['image_id']][b['category_id']].append(b)

    return chunks


def unchunk(boxes):
    result = []
    for key, item in boxes.items():
        for key_cat, item_cat in item.items():
            result.append(item_cat)
    return result


def load_json(path):
    with open(path, 'r') as input:
        boxes = json.load(input)
        return boxes


def save_json(boxes, path):
    with open(path, "w") as out:
        json.dump(boxes, out)


def process_boxes(boxes, min_thresh=0.):
    boxes = chunk_by_image(boxes)

    boxes = filter_top(boxes, min_thresh=min_thresh)
    boxes = unchunk(boxes)
    # boxes = filter_overlap(boxes)

    return boxes


if __name__ == "__main__":
    boxes = sys.argv[1]
    output = sys.argv[2]
    boxes = load_json(boxes)
    boxes = chunk_by_image(boxes)

    boxes = filter_top(boxes)
    boxes = unchunk(boxes)
    boxes = filter_overlap(boxes)
    save_json(boxes, output)
