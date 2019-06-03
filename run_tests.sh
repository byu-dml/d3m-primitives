#! /bin/bash

pip3 install .
mv byudml tmp_byudml
reset
python3 run_tests.py
python3 pipelines/generate_pipelines.py
python3 primitive_jsons/generate_primitive_jsons.py
mv tmp_byudml byudml
pip3 uninstall -y byudml > /dev/null
