import cv2
# from pycocotools.coco import COCO
from pycocotools import mask as cocomask

import numpy as np
import os
# import tensorflow.keras as keras





# matplotlib inline
# from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
# import pylab
# import random
import os
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)


class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            data_directory,
            ids,
            coco,
            classes = None,
            augmentation = None,
            preprocessing = None,
            
    ):        
        self.augmentation = augmentation
        self.class_values = [classes.index(cls.lower()) + 1 for cls in classes]
        self.coco = coco
        self.ids  = ids
        self.IMAGES_DIRECTORY = data_directory
        self.preprocessing    = preprocessing


    def __getitem__(self, index):
        # load image object.
        img = self.coco.loadImgs(self.ids[index])[0]        #info.

        image_path = os.path.join(self.IMAGES_DIRECTORY, img["file_name"]) #path
        image = plt.imread(image_path)
        image = np.array(image)
        image = cv2.resize(image, (608, 608),interpolation = cv2.INTER_LINEAR)


        # load annotation object.
        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])     # segmentation_id into annotation randon id
        annotations = self.coco.loadAnns(annotation_ids)           # annotation_segmentation for segmentation_id


        # Convert segmentation to pixel level masks
        buffer = []
        for i in range(len(annotation_ids)):
            rle = cocomask.frPyObjects(annotations[i]['segmentation'], img['height'], img['width'])

            mask = cocomask.decode(rle)
            mask = mask.flatten()
            try:
                buffer = [mask[i]+buffer[i] for i in range(len(mask))]
            except:
                buffer = mask

        mask = np.array(buffer)
        mask = mask.reshape((img['height'], img['width']))
        mask = cv2.resize(mask, (608, 608), interpolation = cv2.INTER_LINEAR)
        mask_aux = np.zeros_like(mask)
        mask_aux[mask != 0] = 1
        mask_aux = np.expand_dims(mask_aux, axis = 2).astype('float')
        mask = mask_aux
        

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.ids)