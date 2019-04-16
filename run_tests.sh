#! /bin/bash

pip3 install .
mv byudml tmp_byudml
reset
python3 run_tests.py
mv tmp_byudml byudml
pip3 uninstall -y byudml > /dev/null
