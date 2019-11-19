import json
import os

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor
from byudml.aggregator.aggregator import Aggregator
from byudml import (
    __imputer_path__, __metafeature_path__, __aggregator_path__,
    __imputer_version__, __metafeature_version__, __aggregator_version__
)
import sys
sys.path.append(".")
from submission.utils import get_new_d3m_path, clear_directory

PRIMITIVE_JSON = "primitive.json"

def save_primitive_json(primitive, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, 0o777)

    with open(path, 'w') as f:
        print("Writing JSON primitive to", path)
        primitive_json = primitive.metadata.to_json_structure()
        json.dump(primitive_json, f, indent=4, sort_keys=True)
        os.chmod(path, 0o777)



byu_dir = get_new_d3m_path()
clear_directory(byu_dir)
IMPUTER_JSON_PATH = os.path.join(byu_dir, __imputer_path__, __imputer_version__, PRIMITIVE_JSON)
METAFEATURE_JSON_PATH = os.path.join(byu_dir, __metafeature_path__, __metafeature_version__, PRIMITIVE_JSON)
AGGREGATOR_JSON_PATH = os.path.join(byu_dir, __aggregator_path__, __aggregator_version__, PRIMITIVE_JSON)

save_primitive_json(RandomSamplingImputer, IMPUTER_JSON_PATH)
save_primitive_json(MetafeatureExtractor, METAFEATURE_JSON_PATH)
save_primitive_json(Aggregator, AGGREGATOR_JSON_PATH)
