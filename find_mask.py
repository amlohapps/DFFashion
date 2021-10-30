import json
import glob

import cv2

import warnings 
warnings.filterwarnings("ignore")

from mrcnn.config import Config
from utils import to_rle

import mrcnn.model as modellib

IMAGE_SIZE = 512
NUM_CATS = 46

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 100
    

    STEPS_PER_EPOCH = 5500
    VALIDATION_STEPS = 100
    
config = FashionConfig()
# config.display()

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

glob_list = glob.glob('mask_rcnn_fashion_0011.h5')
model_path = glob_list[0] if glob_list else ''

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir='')

assert model_path != '', "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


with open(".data/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]


def masker(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    mask = model.detect([img])
    r = mask[0]
    
    return img, r

def masker_np(img):

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    mask = model.detect([img])
    r = mask[0]
    
    return img, r
