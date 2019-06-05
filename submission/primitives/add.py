#!/usr/bin/env python3

# A simple script to help add a new (or updated) primitive annotation. Provide it with a list
# of paths to primitive annotations and it will move them into the place inside the repository.

import json
import os
import os.path
import sys

for primitive_annotation_path in sys.argv[1:]:
    with open(primitive_annotation_path, 'r') as primitive_annotation_file:
        primitive_annotation = json.load(primitive_annotation_file)

    os.renames(primitive_annotation_path, os.path.join(
        'v' + primitive_annotation['primitive_code']['interfaces_version'],
        primitive_annotation['source']['name'],
        primitive_annotation['python_path'],
        primitive_annotation['version'],
        'primitive.json',
    ))
