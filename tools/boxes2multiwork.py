import argparse


parser = argparse.ArgumentParser(description='Process the bbox.json file and convert detections to the Matlab format that will be placed in multiwork')
parser.add_argument("--boxes_file", type=str, default="input")
parser.add_argument("--out_dir", type=str, default="output")
parser.add_argument("--exp", type=int, default=15)
parser.add_argument("--infer_set", type=bool, default=False)
args = parser.parse_args()