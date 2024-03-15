# --------------------------------------------------------
# Configurations for domain adaptation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# Adapted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py
# --------------------------------------------------------

import os.path as osp

import numpy as np
from easydict import EasyDict

from utils import project_root

# from advent.utils import project_root
# from advent.utils.serialization import yaml_load


cfg = EasyDict()

# Number of object classes
cfg.NUM_CLASSES = 19

# CUDA
cfg.GPU_ID = 0

#directory
cfg.OUTPUR_DIR = str(project_root / 'output_file')
cfg.DATA_DIR = str(project_root / 'data')
cfg.ROOT_DIR = str(project_root)
cfg.INFO = str(project_root / 'dataset/umbc_info.json')

# model
cfg.MULTI_LEVEL = (True,)
cfg.checkpoint = str(project_root / 'checkpoints/model_umbc.pth')
cfg.checkpoint_tuned = str(project_root / 'checkpoints/model_umbc_finetuned.pth')
cfg.BATCH_SIZE = 1
cfg.NUM_WORKERS = 1
cfg.MAX_ITERS = 5000

# model hyperparameter
cfg.LEARNING_RATE = 2.5e-4
cfg.MOMENTUM = 0.9
cfg.WEIGHT_DECAY = 0.0005
cfg.POWER = 0.9
cfg.LAMBDA_SEG_MAIN = 1.0
cfg.LAMBDA_SEG_AUX = 0.1
cfg.EARLY_STOP = 100
cfg.LAMBDA_SEG_MAIN = 1.0
cfg.LAMBDA_SEG_AUX = 0.1




# data
cfg.INPUT_SIZE_SOURCE = (1280, 720)
cfg.INPUT_SIZE = (640, 480)
cfg.OUTPUT_SIZE = (640, 480)
cfg.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)