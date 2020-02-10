import multiprocessing as mp
import os
import sys
import json
import argparse
import postprocess as post

# def partition_subjects(boxes, dataset):
#     subjs = {}
#
#     for img in dataset['images']:
#         print()

def smoothing(boxes, orders, subjs, id2fnum=None, fnum2id=None, window_size=3, adj_thresh=4, ):

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


            # remove detections that are lone islands, with no adjacent detections
            ordered_items = post.filter_islands(ordered_items, adj_thresh)
            # add missing detections, filling in the holes between adjacent detections
            ordered_items = post.add_missing(ordered_items, adj_thresh, id2fnum, fnum2id)

            # dump the resulting detections into the global results
            for b in ordered_items:
                results.append(b[0])

    return results

def process():
    print()


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

    id2fnum, fnum2id = post.id_to_fnum(dataset)

    orders, subjs = post.get_ordering(boxes, dataset)

    mapped_boxes = post.make_mapped_boxes(boxes)

    mapped_boxes = post.filter_duplicates(mapped_boxes)

    # subjs = partition_subjects(boxes, dataset)

    procs = []

    for subj, ids in subjs.items():
        print()




    print()