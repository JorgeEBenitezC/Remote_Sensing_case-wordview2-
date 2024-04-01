import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import multivariate_normal


parser = argparse.ArgumentParser(description='script para filtrar imagens com ruído utilizando um kernel gaussiano.')
parser.add_argument("--data_dir", 
                    type = str, 
                    default = f'{os.getcwd()}/data/raw/questao_1/airport_gray_noisy.PNG', 
                    help = "caminho onde se encontram as imagens em bruto.")


class Image_filter():

    def __init__(self, config):
        self.path_dir = config.data_dir


    def run(self):
        
        img_ruido = cv.imread(self.path_dir)


        k = 10
        tamanho = 2 * k +1
        # kernel = np.ones((tamanho, tamanho), np.float32)/(tamanho**2)


        # kernel gauss.
        mean = [0, 0]
        cov = [[2, 0], [0, 2]]

        kernel = np.zeros((tamanho, tamanho), np.float32)
        for i in range(tamanho):
            for j in range (tamanho):
                x = [ -k+i, -k+j]
                w = multivariate_normal.pdf(x, mean, cov)
                kernel[i][j] = w


        img_filtrada = cv.filter2D(img_ruido, -1, kernel)

        plt.imshow(img_filtrada, cmap = 'gray')
        plt.title('imagem filtrada')
        plt.show()


if __name__ == "__main__":
    config = parser.parse_args()

    filter = Image_filter(config)
    start_create = filter.run()
