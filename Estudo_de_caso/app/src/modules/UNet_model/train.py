import albumentations as A
import cv2
import random
import tensorflow as tf
import time

from app.src.modules.UNet_model.dataloader import Dataloder
from app.src.modules.UNet_model.dataset import Dataset
from app.src.modules.UNet_model.model import Model
from tensorflow.keras import backend as K
from pycocotools.coco import COCO
# from pycocotools import mask as cocomask


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0],'GPU')


class Train():
    def __init__(self, 
                 config
    ):
        
        self.backbone   = config.backbone
        self.batch_size = config.batch_size
        self.classes    = config.classes
        self.data_dir   = config.data_directory
        self.epochs     = config.epochs
        self.flag       = config.flag
        self.model_dir  = config.model_dir
        self.porcent    = config.split
        self.lr         = config.lr

        self.obj_model = Model(self.backbone, self.classes, self.lr)  
        self.UNet      = self.obj_model.create()
        self.model            = self.obj_model.get_Model()
        self.n_classes        = self.obj_model.get_Nclasses()
        self.preprocess_input = self.obj_model.get_PreproInput()
        self.train_dataset = None


    # define heavy augmentations
    def __get_training_augmentation(self):
        train_transform = [
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.Rotate(limit=[-30,30], interpolation=cv2.INTER_CUBIC, p=0.55, border_mode = cv2.BORDER_CONSTANT),
        ]
        return A.Compose(train_transform)


    def __get_preprocessing(self, preprocessing_fn):

        _transform = [
                       A.Lambda(image=preprocessing_fn),
                     ]
        return A.Compose(_transform)
    

    def __ds_construct(self,
                       IMAGES_DIRECTORY,
                       ids,
                       coco
                      ):
        return Dataset( IMAGES_DIRECTORY,
                        ids,
                        coco,
                        classes=self.classes,
                        augmentation=self.__get_training_augmentation(),
                        preprocessing=self.__get_preprocessing(self.preprocess_input),
                       )
    
    
    def run(self): 

        # Split dataset.
        IMAGES_DIRECTORY = f"{self.data_dir}train/images"
        # ANNOTATIONS_PATH = f"{self.data_dir}annotation.json"
        ANNOTATIONS_PATH = f"{self.data_dir}annotation-small.json"
        coco = COCO(ANNOTATIONS_PATH)

        image_ids = coco.getImgIds(catIds=coco.getCatIds())


        # ------------------------------------------------------
        id_para_provar_fijos = [149691, 247999, 51403, 149714, 
                                51413, 182496, 116964, 84202, 
                                215281, 84217, 182523, 116988, 
                                215295, 182534, 149777, 248085, 
                                248090, 117022, 182561,]
        for id_remove in id_para_provar_fijos:
            image_ids.remove(id_remove)
        # ------------------------------------------------------
    

        k = int(len(image_ids)*(self.porcent[1]/100)) 
        small_ds = random.SystemRandom().sample(image_ids, k)
        [image_ids.remove(id) for id in small_ds]

        valid_ids, test_ids = [small_ds[i::2] for i in range(2)]
                

        # Dataset cosntruct.
        train_dataset = self.__ds_construct(IMAGES_DIRECTORY, image_ids, coco)
        valid_dataset = self.__ds_construct(IMAGES_DIRECTORY, valid_ids, coco)
        test_dataset  = self.__ds_construct(IMAGES_DIRECTORY, test_ids, coco)

        train_dataloader = Dataloder(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)
        test_dataloader  = Dataloder(test_dataset, batch_size=1, shuffle=False)


        # check shapes for errors
        assert train_dataloader[0][0].shape == (self.batch_size, 608, 608, 3)
        assert valid_dataloader[0][1].shape == (1, 608, 608, self.n_classes) 
        assert test_dataloader[0][1].shape == (1, 608, 608, self.n_classes) 

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{self.model_dir}best_model',
                                                        monitor='val_iou_score',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='max',
                                                        save_freq = 'epoch')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor = tf.math.exp(-0.1),
            patience = 5,
            verbose=1,
            mode='auto',
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,mode='auto',
                                                    #min_delta=0.001
                                                    )

        # Log tensor board
        log_dir = f'{self.model_dir}logs/'
        callback_tensorbord = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

        callbacks = [
                    early_stop,
                    checkpoint,
                    reduce_lr,
                    callback_tensorbord,
                    ]
        

        # fit model.
        start = time.time()
        history = self.model.fit(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=valid_dataloader,
            validation_steps=len(valid_dataloader),
        )        
        end = time.time()
        print('\ntime fit model', end - start,'\n')

        n_epochs = len(history.history['loss'])
                
        
        # re-fit model.
        if self.flag == 1:
            print('\n\nStaring Re-Fit...')
            self. model.optimizer.lr

            # Change larning rate
            K.set_value(self.model.optimizer.learning_rate, self.lr /10)
            self.model.optimizer.lr

            # re-fit model
            start = time.time()
            history = self.model.fit(train_dataloader,
                initial_epoch = n_epochs,
                epochs = n_epochs + self.epochs,
                steps_per_epoch=len(train_dataloader),
                validation_data=valid_dataloader,
                validation_steps=len(valid_dataloader),
                verbose=1,
                callbacks=callbacks
            )
            end = time.time()
            print('\n time fit model', end - start,'\n')

        
        
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries
        def visualize(image_size = (16,5),**images):
            """PLot images in one row."""
            n = len(images)
            plt.figure(figsize=image_size)
            for i, (name, image) in enumerate(images.items()):
                plt.subplot(1, n, i + 1)
                plt.xticks([])
                plt.yticks([])
                plt.title(' '.join(name.split('_')).title())
                plt.imshow(image)
            plt.show()


        import numpy as np
        n = 4
        ids = np.random.choice(np.arange(len(test_dataloader)), size=n)

        for i in ids:
            image, gt_mask = test_dataset[i]
            i_model = np.expand_dims(image, axis=0)
            pr_mask = self.model.predict(i_model)
            pr_mask = (pr_mask > 0.5).astype(np.uint8)
        
            visualize(
                image= image,
                gt_mask=gt_mask,
                pr_mask=pr_mask.squeeze(),
                pr_mask1 = mark_boundaries(image, pr_mask.squeeze())
            )