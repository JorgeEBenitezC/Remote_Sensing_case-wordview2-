import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm

from segmentation_models import Unet
from tensorflow import keras

sm.set_framework('tf.keras')
sm.framework()


class Model:
    def __init__ (self, 
                  backbone, 
                  classes, 
                  lr
    ):
        
        self.__backbone = backbone         
        self.__classes = classes
        self.__lr = lr
        self.__model = None
        self.__n_classes = None
        self.__preprocess_input = None

    
    def create(self):
        self.__preprocess_input = sm.get_preprocessing(self.__backbone)


        # define network parameters
        self.__n_classes = 1 if len(self.__classes) == 1 else (len(self.__classes) + 1)  # case for binary and multiclass segmentation
        self.activation = 'sigmoid' if self.__n_classes == 1 else 'softmax'


        self.__model = sm.Unet(self.__backbone, classes=self.__n_classes,  
        #                 encoder_weights='imagenet', 
                        encoder_weights=None,
                        input_shape=(608, 608, 3), 
                        activation=self.activation
                    )


        # define optomizer
        self.optim = keras.optimizers.Adam(self.__lr)


        # define metrics.
        self.metrics     = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        self.jaccar_loss = sm.losses.JaccardLoss()
        self.total_loss  =  self.jaccar_loss # + (1 * BiorCat_loss)


        # compile keras model with defined optimozer, loss and metrics
        self.__model.compile(self.optim,  self.total_loss,  self.metrics)

        return self.__model, self.__n_classes, self.__preprocess_input
    
    
    def get_Model(self):
        return self.__model
    

    def get_Nclasses(self):
        return self.__n_classes
    

    def get_PreproInput(self):
        return self.__preprocess_input