FROM python:3-onbuild

RUN mkdir -p /workdir
WORKDIR /workdir
COPY requirements.txt /workdir


RUN pip install -r requirements.txt
