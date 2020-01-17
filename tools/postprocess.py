import numpy as np
import json
import argparse
from copy import deepcopy


def filter_low_confidence(boxes, conf=0.80):
    new_boxes = []

    for b in boxes:
        if b['score'] < conf:
            continue
        new_boxes.append(b)

    return new_boxes

def filter_duplicates(mapped_boxes):

    new_mapped_boxes = {}

    for img, boxes in mapped_boxes.items():
        items = set([b['category_id'] for b in boxes])
        new_mapped_boxes[img] = []
        for i in items:
            instances = [b for b in boxes if b['category_id'] == i]
            instances = sorted(instances, key=lambda x: x['score'], reverse=True) # sort by decreasing score
            new_mapped_boxes[img].append(instances[0]) # take the top scoring

    return new_mapped_boxes


def get_ordering(boxes, dataset):

    orders = {}
    subjs = {}

    for img in dataset['images']:
        name = img['file_name']
        sname = name.split("_")
        subj = sname[1] + sname[2]
        id = img['id']

        fnum = int(sname[3].replace(".jpg", "").replace("frame", ""))

        if subj not in subjs:
            subjs[subj] = []

        subjs[subj].append(id)

        orders[id] = fnum

    return orders, subjs


def smoothing(boxes, orders, subjs, window_size=2, adj_thresh=3):

    results = []

    for subj, imgs in subjs.items():

        img_ordered = [(i, orders[i]) for i in imgs]
        img_ordered = sorted(img_ordered, key=lambda x: x[1])


        for item in range(1, 13):

            item_boxes = []

            # pull out all the box detections for `item`
            for i in img_ordered:
                if i[0] not in boxes:
                    continue
                for b in boxes[i[0]]:
                    if b['category_id'] == item:
                        item_boxes.append(b)

            ordered_items = sorted([(b, orders[b['image_id']])
                                    for b in item_boxes],
                                    key=lambda x: x[1])

            if len(ordered_items) <= 2 * window_size + 1:
                for i in ordered_items:
                    results.append(i[0])
                continue

            i = 0
            j = window_size + 1 + window_size

            # remove detections that are lone islands, with no adjacent detections
            ordered_items = filter_islands(ordered_items, adj_thresh)
            # add missing detections, filling in the holes between adjacent detections
            ordered_items = add_missing(ordered_items, adj_thresh)

            # dump the resulting detections into the global results
            for b in ordered_items:
                results.append(b[0])

    return results



def filter_islands(ordered_items, adj_thresh):
    results = []

    for i in range(len(ordered_items)):

        if i == len(ordered_items) - 1:
            results.append(ordered_items[i])
            continue


        # check for a lone detection,
        if ordered_items[i][1] - ordered_items[i - 1][1] > adj_thresh and \
            ordered_items[i + 1][1] - ordered_items[i][1] > adj_thresh:
            continue
        else:
            results.append(ordered_items[i])

    return results

def add_missing(items, adj_thresh=3, num_neighbs=2, comp_thresh=2):

    i = 0
    while i < len(items):

        # handle beginning
        if i < num_neighbs:
            i += 1
            continue
        # handle end
        if i >= len(items) - num_neighbs:
            i += 1
            break

        fwd_neighbs = sorted([items[i + j] for j in range(1, num_neighbs+1)], key=lambda x: x[1])
        back_neighbs = sorted([items[i - j] for j in range(1, num_neighbs+1)], key=lambda x: x[1])

        fsum = sum(fwd_neighbs[i][1] - fwd_neighbs[i-1][1] for i in range(1, len(fwd_neighbs)))
        bsum = sum(back_neighbs[i][1] - back_neighbs[i - 1][1] for i in range(1, len(back_neighbs)))

        # check if detections in this time window are compact
        compact_neighb = False
        if fsum < (num_neighbs - 1 + comp_thresh) and bsum < (num_neighbs - 1 + comp_thresh):
            compact_neighb = True

        bdiff = items[i][1] - back_neighbs[-1][1]
        fdiff = fwd_neighbs[0][1] - items[i][1]


        if bdiff > 1 and bdiff <= adj_thresh and\
            fdiff >= 1 and fdiff <= adj_thresh and\
                compact_neighb:

            for j in range(bdiff - 1):
                new_det = (deepcopy(back_neighbs[-1]), back_neighbs[-1][1] + j + 1)
                items.insert(i, new_det)
                i += 1

            i += 1
            continue
        else:
            i += 1
            continue

    for i, x in enumerate(items):
        if i > 0:
            if items[i-1][1] > items[i][1]:
                raise Exception("ordering failure")

    return items

def make_mapped_boxes(boxes):

    img_boxes = {}

    for b in boxes:
        id = b['image_id']

        if id not in img_boxes:
            img_boxes[id] = []

        img_boxes[id].append(b)

    return img_boxes



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Postprocess object localization results')
    parser.add_argument("--bbox_file", type=str)
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    with open(args.bbox_file, "r") as input:
        boxes = json.load(input)

    with open(args.dataset_file, "r") as input:
        dataset = json.load(input)


    boxes = filter_low_confidence(boxes)

    orders, subjs = get_ordering(boxes, dataset)

    mapped_boxes = make_mapped_boxes(boxes)

    mapped_boxes = filter_duplicates(mapped_boxes)

    results = smoothing(mapped_boxes, orders, subjs)

    with open(args.output, "w") as out:
        json.dump(results, out)