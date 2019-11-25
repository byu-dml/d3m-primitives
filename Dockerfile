FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10
ADD . /d3m-primitives
RUN pip3 install -r /d3m-primitives/requirements.txt
WORKDIR /d3m-primitives
