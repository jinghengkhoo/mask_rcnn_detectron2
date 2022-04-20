from detectron2.utils.logger import setup_logger
from torch import device

setup_logger()

import os
import pickle

from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor

from utils import get_train_cfg, plot_samples, ValidationLoss

#config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#checkpoint_url =  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
checkpoint_url =  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

output_dir = "./output"

#class_names = ['bed','bench','boat','chair','couch', 'dining table','refrigerator',
#                'skateboard','skis','snowboard','suitcase','surfboard','tv', 'vase']

class_names = ["bench", "suitcase", "tv", "refrigerator", "vase", "table", "skis", "snowboard", "skateboard", "bed",
                "couch", "boat", "surfboard", "chair", "Canoe", "Chest of drawers", "Christmas tree", "Jet ski", "Piano",
                 "Picture frame", "Punching bag", "Sculpture", "Whiteboard"]

num_classes = len(class_names)

device = "cuda"

train_ds_name = "dummy_Train"
train_images_path = "/home/ubuntu/RoomScan/CombinedDataset/TotalTrainImages"
train_json_path = "/home/ubuntu/RoomScan/CombinedDataset/merge_train.json"

val_ds_name = "dummy_Val"
val_images_path = "/home/ubuntu/RoomScan/CombinedDataset/TotalValImages"
val_json_path = "/home/ubuntu/RoomScan/CombinedDataset/merge_val.json"

cfg_save_path = "SEG_cfg.pickle"

register_coco_instances(name = train_ds_name, metadata={},
json_file=train_json_path, image_root=train_images_path)

register_coco_instances(name = val_ds_name, metadata={},
json_file=val_json_path, image_root=val_images_path)

plot_samples(train_ds_name, n=2)

def main():
    
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_ds_name, val_ds_name, num_classes, 
                    device, output_dir, num_workers = 4, iterations=5000)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    ## Training ##
    
    trainer = DefaultTrainer(cfg) 
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    ## Evaluation ##
    with open(cfg_save_path, 'rb') as f:
        eval_cfg = pickle.load(f)
    eval_cfg.MODEL.WEIGHTS = os.path.join(eval_cfg.OUTPUT_DIR, "model_final.pth")
    eval_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    #predictor = DefaultPredictor(eval_cfg)

    evaluator = COCOEvaluator(val_ds_name, eval_cfg, False, output_dir="./eval/")
    val_loader = build_detection_test_loader(eval_cfg, val_ds_name)
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    main()




