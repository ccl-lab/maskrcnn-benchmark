#python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/c4_latest_green/inference/home_15_green_sample/bbox.json --data ../datasets/home/annotations/test15_subsample_green.json --out_dir training/c4_latest_green/boxed_images/
#python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/c4_latest_black/inference/home_15_black_sample/bbox.json --data ../datasets/home/annotations/test15_subsample_black.json --out_dir training/c4_latest_black/boxed_images/
#python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/c4_latest_green/inference/home_15_test_green/bbox.json --data ../datasets/home/annotations/test15_subsample_green.json --out_dir training/c4_latest_green/boxed_images/
#python add_boxes.py --image_dir ../datasets/home/train15 --boxes training/c4_latest/inference/home_15_test/bbox.json --data ../datasets/home/annotations/test15.json --out_dir training/c4_latest/boxed_images/
#python add_boxes.py --image_dir ../datasets/home/train15 --boxes training/c4_latest_black/inference/home_15_test_black/bbox.json --data ../datasets/home/annotations/test15_green.json --out_dir training/c4_latest_black/boxed_images/
#python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/c4_latest_green/inference/home_15_green_sample/bbox.json --data ../datasets/home/annotations/test15_subsample_green.json --out_dir training/c4_latest_green/boxed_images
#python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/c4_latest_nopot/inference/home_15_green_sample/bbox.json --data ../datasets/home/annotations/test15_subsample_green.json --out_dir training/c4_latest_nopot/boxed_images
python add_boxes.py --image_dir ../datasets/home/infer15 --boxes training/FPN_latest_multipot/inference/home_15_green_sample/bbox.json --data ../datasets/home/annotations/test15_subsample_green.json --out_dir training/FPN_latest_multipot/boxed_images