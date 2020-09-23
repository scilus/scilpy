From python:3.7

RUN apt-get update
RUN apt-get -y upgrade

RUN apt -y install libblas-dev
RUN apt -y install liblapack-dev
RUN apt -y install libgl1-mesa-glx

WORKDIR /
ENV SCILPY_VERSION="master"
RUN wget https://github.com/scilus/scilpy/archive/${SCILPY_VERSION}.zip
RUN unzip ${SCILPY_VERSION}.zip
RUN mv scilpy-${SCILPY_VERSION} scilpy

WORKDIR /scilpy
RUN pip install -e .
