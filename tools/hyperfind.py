import torch
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser(description='Find the best model')
parser.add_argument("--start_dir", type=str, default="hypertune")
args = parser.parse_args()

results = []

for root, dirs, files in os.walk(args.start_dir):
    for dir in dirs:
        lr = float(dir)
        p = osp.join(root, dir, "inference",
                     "home_15_test", "coco_results.pth")
        if os.path.exists(p):
            data = torch.load(p)
            vals = list(data.results['bbox'].values())
            avg_vals = sum(vals) / len(vals)
            results.append((round(lr, 5), avg_vals))
    break

results = sorted(results, key=lambda x: x[1], reverse=True)
print()
