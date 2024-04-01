import os


BACKBONE = 'resnet50'
BATCH_SIZE = 1
CLASSES = ['vegetação']
EPOCHS = 25
LR = 0.001
Re_FIT = 0
SPLIT = [70, 30]

DATA_DIR   = f'{os.getcwd()}/data/raw/questao_3/'
MODEL_DIR  = f'{os.getcwd()}/models/{BACKBONE}_320/'
MODEL_LOAD = f'{MODEL_DIR}best_model/'
