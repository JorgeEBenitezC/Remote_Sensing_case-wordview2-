FROM python:3.10.12
WORKDIR /usr/src

RUN git clone https://github.com/JorgeEBenitezC/Remote_Sensing_case-wordview2-.git
RUN pip install -r Remote_Sensing_case-wordview2-/Estudo_de_caso/requeriments.txt
RUN python -m pip install -q tensorflow==2.2.1
RUN pip install -U -q segmentation-models
RUN pip install -q keras==2.5
CMD ["bash"] 
