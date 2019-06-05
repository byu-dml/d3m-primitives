#!/usr/bin/env python3

# A simple script to help disable a failing primitive annotation. It moves a primitive annotation
# to "failed" directory. The script can be called in two ways:
#
#  - with primitive ID as reported by the validation script (e.g., v2018.1.26/d3m.primitives.dsbox.MultiTableFeaturization/0.1.3)
#  - with path to failing primitive annotation file (e.g., ./v2018.1.26/ISI/d3m.primitives.dsbox.RandomProjectionTimeSeriesFeaturization/0.1.3/primitive.json)

import argparse
import glob
import os
import os.path
import shutil

parser = argparse.ArgumentParser(description="Disable primitives by moving them to 'failed' directory.")
parser.add_argument('primitive_names', metavar='primitive', nargs='+', help="primitive name to disable")
arguments = parser.parse_args()

for primitive_name in arguments.primitive_names:
    if os.path.exists(primitive_name):
        segments = primitive_name.split('/')
        interface_version, python_path, version, filename = [segments[i] for i in [-5, -3, -2, -1]]
        if filename != 'primitive.json':
            raise ValueError("Not a path to \"primitive.json\" file: " + primitive_name)
    else:
        interface_version, python_path, version = primitive_name.split('/')

    for globbed_file_path in glob.glob('{interface_version}/*/{python_path}/{version}'.format(interface_version=interface_version, python_path=python_path, version=version)):
        path = os.path.join('failed', globbed_file_path)
        shutil.rmtree(path, ignore_errors=True)
        os.renames(globbed_file_path, path)
