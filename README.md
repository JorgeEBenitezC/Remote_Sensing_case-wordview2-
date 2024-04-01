# Remote sensing estudo de caso.
O principal objetivo deste repositório é mostrar uma série de técnicas de visão computacional para o pré-processamento, 
consumo e pós-processamento de imagens multiespectrais do satélite worldview 2, buscando a segmentação de pixels em diferentes cenários.

O processamento de imagens foi realizado por meio de técnicas clássicas de visão computacional e modelos de aprendizagem profunda (Unet).

A execução de todos os scripts nesse repositório foi feita em um ambiente virtual python, alem espera-se que a estrutura de pastas seja a mostrada abaixo.

O repositório tem um arquivo "Dockerflie" para a virtualização do ambiente via Docker.
Use o seguinte comando para executá-lo:
    
    docker build -t 'nome_do_seu_contêiner_aqui' .

## Installation
### Ambiente virtual
dependências:

    Python==3.10.12
    albumentations==1.3.1 
    geopandas==0.14.3
    matplotlib==3.7.1  
    opencv-python==4.8.0.76 
    pycocotools==2.0.7
    rasterio==1.3.9
    segmentation-models==1.0.1
    scikit-image==0.19.3
    tensorflow==2.11.0

## Estrutura de Pastas:
app: Diretório que contém todos os códigos de visão computacional;  
src: Diretório que contém os principais arquivos fonte da solução;  
modules: Diretório que contém os módulos utilizados;  
utils: Diretório que contém funções que podem ser utilizadas de maneira isolada;  
data: Diretório que contém os arquivos que podem ser utilizados como entrada e os arquivos de saída;  
data_augmentation: Diretório que contém os arquivos do conjunto de dados de "aumentados";  
processed: Diretório que contém os arquivos do conjunto de dados de forma "processada";  
raw: Diretório que contém os arquivos do conjunto de dados de forma "bruta";  
manual: Diretório que contém os manuais do sistema;  
readme: Diretório que contém os arquivos complementares utilizados no README do projeto;  
models: Diretório que contém os modelos de visão computacional;

# Images processing
## Salt & Pepper noise reduction.
    python Filtro_gaussiano.py  --data_dir='your_path_here'

![img_1_GH](https://github.com/JorgeEBenitezC/Remote_Sensing_case-wordview2-/assets/164698211/2b17e7b8-7a26-4329-a584-43937e5b5d54)

## RGBimage segmentation.
    python HSV_segmentation.py --data_dir='your_path_here'
    
![img_2_HB](https://github.com/JorgeEBenitezC/Remote_Sensing_case-wordview2-/assets/164698211/ccea159d-f752-4c5b-aead-91aad8e8fbf8)

    
## Raster segmentation.
    python NDVI_segmentation.py --data_dir='your_path_here'
    
![img3_HB](https://github.com/JorgeEBenitezC/Remote_Sensing_case-wordview2-/assets/164698211/fd8663f5-75ef-47e6-9624-9d664e3faca5)


## deep_learning segmentation.
    python main.py 
    
![img_4_HB](https://github.com/JorgeEBenitezC/Remote_Sensing_case-wordview2-/assets/164698211/3af51f28-45b6-45f1-9869-daabc662860c)
