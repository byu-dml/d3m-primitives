FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2019.2.18
ADD . /d3m-primitives
WORKDIR /d3m-primitives
RUN pip3 install -e .
