import argparse
import cv2 as cv
import numpy as np
import os
import rasterio


parser = argparse.ArgumentParser(description='Script para segmentação de vegetação utilizando NDVI e métodos clássicos.')
parser.add_argument("--data_dir", 
                    type = str, 
                    default = f'{os.getcwd()}/data/raw/questao_2b/', 
                    help = "caminho onde se encontram as imagens (raster) em bruto.")


class Vegetation_segmentation():

    def __init__(self, config):
        self.path_dir = config.data_dir


    def __georeference(self, 
                       item_1, 
                       item_2, 
                       item_3, 
                       item_4, 
                       item_5
                       ):
         
        arr = item_1
        profile = item_2
        path = item_3
        name = item_4
        id = item_5

        arr = np.reshape(arr, (1, arr.shape[0], arr.shape[1])) 

        outMeta = { 'driver': 'GTiff', 
                    'dtype': 'float32', 
                    'nodata': 0, 
                    'width': profile['width'],         
                    'height': profile['height'],         
                    'count': 1, 
                    'crs': profile['crs'],
                    'transform': profile['transform'], 
                    'tiled': False, 
                    'interleave':'band'
                }
        
        with rasterio.open(f'{path}{name}_r_{id}.tif', 'w', **outMeta) as m:
            m.write(arr)


    def __post_processing(self, 
                       item_r
                       ):
        arr = item_r

        # threshold
        arr = np.ma.masked_where((arr>0.5), arr)
        arr = arr.mask.astype(np.uint8)

        # kernelS
        kernel1 = np.ones((7, 7), np.uint8)        
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)) 

        # fechamento.
        arr = cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel2)
        # arr = cv.morphologyEx(arr, cv.MORPH_OPEN, kernel2)
        return arr
    

    def run(self):
        out_dir =  f'{os.getcwd()}/data/processed/'
        rasters_paths = sorted([ f.path for f in os.scandir(self.path_dir) if f.is_file() ])
        
        for id, raster_path in enumerate(rasters_paths):
            with rasterio.open(raster_path) as raster: 

                profile = raster.profile
                arr = raster.read()

                
                blue, green, red, = arr[1, :, :], arr[2, :, :], arr[4, :, :]
                NIR1, NIR2 = arr[6, :, :], arr[7, :, :]


                # Allow division by zero
                np.seterr(divide='ignore', invalid='ignore')

    
                # Composições espectrais.
                NDVI = (NIR2.astype(float)-red.astype(float)) / (NIR2.astype(float)+red.astype(float))
                # GC1 = (NIR1.astype(float)/green.astype(float))-1
                # EVI = ((NIR1.astype(float)-red.astype(float)) / (NIR1.astype(float) + (red.astype(float)*6) - (blue.astype(float)*7.5) + 1)) * 2.5


                indice = NDVI
                indice = self.__post_processing(indice)
                self.__georeference(indice, profile, out_dir, 'NDVI', id)
   

if __name__ == "__main__":
    config = parser.parse_args()

    segmentation = Vegetation_segmentation(config)
    start_create = segmentation.run()