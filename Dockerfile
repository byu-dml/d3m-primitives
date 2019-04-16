FROM registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.4.4
ADD . /d3m-primitives
WORKDIR /d3m-primitives
