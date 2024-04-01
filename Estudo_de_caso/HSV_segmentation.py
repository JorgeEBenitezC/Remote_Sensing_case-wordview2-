import argparse
import cv2 as cv
import numpy as np
import os


parser = argparse.ArgumentParser(description='Script para segmentação de aeroporto utilizando HSV e métodos clássicos.')
parser.add_argument("--data_dir", 
                    type = str, 
                    default = f'{os.getcwd()}/data/raw/airport.PNG', 
                    help = "caminho onde se encontram as imagens em bruto.")
parser.add_argument("--l_cores", 
                    type = int, 
                    default = [12, 70, 155], 
                    help = "limite inferior de cor.")
parser.add_argument("--u_cores", 
                    type = int, 
                    default = [17, 120, 200], 
                    help = "limite superior de cor.")

class Vegetation_segmentation():

    def __init__(self, config):
        self.path_dir = config.data_dir
        self.l_cores = config.l_cores
        self.u_cores = config.u_cores


    def run(self):
        
        img = cv.imread(self.path_dir)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        lower_color = np.array(self.l_cores)
        upper_color =  np.array(self.u_cores )
        mask = cv.inRange(img_hsv, lower_color, upper_color)

        img_f = mask
        size = 3

        kernel1 = np.ones((size, size), np.uint8)        
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))     
        kernel = kernel1

        img_e = cv.erode(img_f, kernel, iterations=2)
        img_d = cv.dilate(img_e, kernel, iterations=10)        

        res = cv.bitwise_and(img, img, mask=img_d)
        cv.imshow('Color', res)

        cv.waitKey(0) 
   

if __name__ == "__main__":
    config = parser.parse_args()

    segmentation = Vegetation_segmentation(config)
    start_create = segmentation.run()