## Use the custom trained segmentation model for inference

import os
import pickle

from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from tqdm import tqdm

from utils import *

cfg_save_path = "SEG_cfg.pickle"

train_ds_name = "dummy_Train"
train_images_path = "/home/ubuntu/RoomScan/CombinedDataset/TotalTrainImages"
train_json_path = "/home/ubuntu/RoomScan/CombinedDataset/merege_train.json"

val_ds_name = "dummy_Val"
val_images_path = "/home/ubuntu/RoomScan/CombinedDataset/TotalValImages"
val_json_path = "/home/ubuntu/RoomScan/CombinedDataset/merege_val.json"

register_coco_instances(name = train_ds_name, metadata={},
json_file=train_json_path, image_root=train_images_path)

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.60

predictor = DefaultPredictor(cfg)

path_1 = "/home/ubuntu/RoomScan/test_images_2/"
path_2 = "/home/ubuntu/RoomScan/test_im/"

for img in tqdm(os.listdir(path_2)):

    test_image_path = path_1 + img
    predict_image(test_image_path, predictor, cfg)