FROM python:3.9-slim

WORKDIR /cocos
RUN mkdir /cocos/results
RUN mkdir /cocos/datasets

COPY ./requirements.txt /cocos
COPY ./train.py /cocos

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# command to be run when the docker container is started
CMD ["python3", "/cocos/train.py"]