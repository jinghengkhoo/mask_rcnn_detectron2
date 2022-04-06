import os
import random

import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import ColorMode, Visualizer


def plot_samples(dataset_name, n = 1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for index, sample in enumerate(random.sample(dataset_custom, n)):
        print(sample["file_name"])
        image = cv2.imread(sample["file_name"])
        v = Visualizer(image[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(sample)
        plt.figure(frameon=False)
        plt.imshow(v.get_image())
        plt.axis('off')
        plt.savefig(f'./Plots/{index}.png', dpi = 200, bbox_inches='tight', pad_inches=0)

def get_train_cfg(cfg_file_path, checkpoint_url, train_ds_name, test_ds_name, num_classes, 
                    device, output_dir, num_workers = 2, iterations=100):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo
    
    cfg.DATASETS.TRAIN = (train_ds_name,)
    cfg.DATASETS.TEST = (test_ds_name,)
    
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = 4
    
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = iterations    
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  
    
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def predict_image(image_path, predictor, cfg=None):

    name = os.path.basename(image_path) 
    img = cv2.imread(image_path)
    outputs = predictor(img)
    
    if cfg:
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=["bench", "suitcase", "tv", "refrigerator", "vase", "table", "skis", "snowboard", "skateboard", "bed",
                "couch", "boat", "surfboard", "chair", "Canoe", "Chest of drawers", "Christmas tree", "Jet ski", "Piano",
                 "Picture frame", "Punching bag", "Sculpture", "Whiteboard"])

    else:
        metadata = {}

    v = Visualizer(img[:,:,::-1], metadata=metadata, 
                    scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(frameon=False)
    plt.imshow(v.get_image())
    plt.axis('off')
    plt.savefig(f'./Results/{name}', dpi = 200, bbox_inches='tight', pad_inches=0)

    
def predict_video(video_path, predictor):
    pass
    # TO Implement


"""["bench","suitcase", "tv", "refrigerator", "vase", "dining table", "skis", "snowboard",
          "skateboard", "bed", "couch", "boat", "surfboard", "chair", 'Canoe', 'Chest of drawers',
           'Christmas tree', 'Jet ski', 'Kettle', 'Loveseat', 'Piano', 'Picture frame',
            'Punching bag', 'Sculpture', 'Sofa bed', 'Whiteboard'])"""