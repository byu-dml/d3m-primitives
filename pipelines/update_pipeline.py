import json
import os
from bson import json_util

from d3m import index as d3m_index
from d3m.metadata import (
    base as metadata_base, pipeline as pipeline_module
)

FILENAME = "random_sampling_imputer/0e5b9a6e-df51-4486-914a-8d862431b81d.json"
# FILENAME = "checkPipeline.json"

""""
DELETE THIS FILE
USED TO SHOW YALL HOW THIS WORKS WHILE I'M GONE
run `python3 update_pipeline.py`, it uses the FILENAME defined above
"""

def update_digest(
    output_filepath, pipeline_json_structure=None, filename=None
):
    """
    This function updates the pipeline's digests and version numbers

    Parameters
    ----------
    output_filepath: the filepath of where to write the new, updated pipeline
    pipeline_json_structure: the pipeline in JSON form.  This or the `filename` parameter is mandatory
    filename: the filename of the pipeline json, so we can read it in
    """
    if pipeline_json_structure is None and filename is None:
        raise Exception
    elif pipeline_json_structure is None:
        with open(filename, "r") as file:
            # reading this in should update it, but we'll check just in cases
            # NOTE: must be a pipeline with no digests, or recent digests
            # NOTE: reading this in as straight JSON doesn't work so we have to use the pipeline_module
            pipeline_to_run = pipeline_module.Pipeline.from_json(string_or_file=file).to_json_structure()
    else:
        pipeline_to_run = pipeline_json_structure

    for step in pipeline_to_run['steps']:
        # if not updated, check and update
        primitive = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                step["primitive"]["python_path"]
            )
        )
        check_step = primitive.to_json_structure()
        # lets verify that both are updated
        assert(check_step["primitive"]["version"] == step["primitive"]["version"])
        assert(check_step["primitive"]["digest"] == step["primitive"]["digest"])

    with open(output_filepath, "w") as f:
        f.write(json.dumps(pipeline_to_run, indent=2, default=json_util.default))


pipeline = update_digest("checkPipeline.json", filename=FILENAME)