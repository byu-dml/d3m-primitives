import json
import os

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor


IMPUTER_JSON_PATH = './primitive_jsons/imputer/primitive.json'
METAFEATURE_JSON_PATH = './primitive_jsons/metafeature_extractor/primitive.json'


def save_primitive_json(primitive, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, 0o777)

    with open(path, 'w') as f:
        primitive_json = primitive.metadata.to_json_structure()
        json.dump(primitive_json, f, indent=4, sort_keys=True)

    os.chmod(path, 0o777)


save_primitive_json(RandomSamplingImputer, IMPUTER_JSON_PATH)
save_primitive_json(MetafeatureExtractor, METAFEATURE_JSON_PATH)
