import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "deeplab_resnet101"
_C.MODEL.NUM_CLASSES = 19

_C.MODEL.CONTRAST = CN()
_C.MODEL.CONTRAST.TAU = 100.0
_C.MODEL.CONTRAST.MOMENTUM = 0.9999

_C.INPUT = CN()
_C.INPUT.IGNORE_LABEL = 255

_C.SOLVER = CN()
# Hyper-parameter
_C.SOLVER.MULTI_LEVEL = True
# constant threshold for target mask
_C.SOLVER.DELTA = 0.9
# weight of feature level contrastive loss
_C.SOLVER.LAMBDA_FEAT = 1.0
# weight of output level contrastive loss
_C.SOLVER.LAMBDA_OUT = 1.0
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.CV_DIR = ""