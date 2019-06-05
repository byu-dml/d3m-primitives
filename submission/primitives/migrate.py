#!/usr/bin/env python3

# This is a helper Python script. See "migrate.sh" for the main script to call.
#
# This helper is just changing the interface version string in JSON. This simple approach might
# fail though if other aspects of a primitive annotation changed (like docstrings).
# In that case you might want to update this script to add do more changes when migrating.

import json
import sys

to_version = sys.argv[1]

for filename in sys.argv[2:]:
    with open(filename, 'r') as file:
        primitive_annotation = json.load(file)

    primitive_annotation['primitive_code']['interfaces_version'] = to_version\

    with open(filename, 'w') as file:
        json.dump(primitive_annotation, file, indent=4)
