FROM tensorflow/tensorflow:1.13.2-gpu-py3-jupyter

# Install package in python
WORKDIR /opt/backend
ADD requirements.txt /opt/backend/requirements.txt

RUN pip install pip --upgrade && \
    pip install -r requirements.txt