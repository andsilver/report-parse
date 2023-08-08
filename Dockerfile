FROM ubuntu:20.04

WORKDIR extract-info

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --upgrade pip

COPY requirements.txt /extract-info/
RUN pip3 install -r requirements.txt

RUN apt-get install -y libgl1
ENV TZ=Asia/Singapore
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libglib2.0-0