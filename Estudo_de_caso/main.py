import argparse

from app.src.modules.UNet_model.train import Train
from config import BACKBONE, BATCH_SIZE, CLASSES, DATA_DIR, EPOCHS, MODEL_DIR, LR, MODEL_DIR, Re_FIT, SPLIT


parser = argparse.ArgumentParser(description='Falta hacer la descripcion')
parser.add_argument("--backbone", 
                    type = str, 
                    default = BACKBONE, 
                    help = "")
parser.add_argument("--batch_size", 
                    type = int, 
                    default = BATCH_SIZE, 
                    help = "")
parser.add_argument("--classes", 
                    type = str, 
                    default = CLASSES, 
                    help = "")
parser.add_argument("--data_directory", 
                    type = str, 
                    default = DATA_DIR, 
                    help = "")
parser.add_argument("--epochs", 
                    type = int, 
                    default = EPOCHS, 
                    help = "")    
parser.add_argument("--model_dir", 
                    type = str, 
                    default = MODEL_DIR, 
                    help = "") 
parser.add_argument("--lr", 
                    type = int, 
                    default = LR, 
                    help = "")    
parser.add_argument("--flag", 
                    type = int, 
                    default = Re_FIT, 
                    help = "")    
parser.add_argument("--split", 
                    type = int, 
                    default = SPLIT, 
                    help = "") 


if __name__ == "__main__":

    config = parser.parse_args()
    object = Train(config)

    print('\nBegin training....')
    start  = object.run()