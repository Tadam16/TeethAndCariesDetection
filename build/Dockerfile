FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update
RUN apt-get install unzip gcc -y

COPY build/floating_requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

RUN pip freeze # for freezing requirements.txt