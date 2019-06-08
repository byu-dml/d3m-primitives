import json
import os
import pymongo
import pandas as pd
from d3m import index as d3m_index
from d3m.metadata import (
    base as metadata_base, pipeline as pipeline_module
)

from byudml import __imputer_path__
from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor
import sys
sys.path.append(".")
from submission.utils import get_new_d3m_path,  extract_byu_info, clear_directory, create_and_add_to_directory, seed_datasets_exlines, create_meta_script_seed

real_mongo_port=12345
lab_hostname = "computer"


def generate_imputer_pipeline(task_type):
    if task_type == 'classification':
        pipeline_id = 'f4fe3fcc-45fe-4c85-8845-549e2f466f21'
    elif task_type == 'regression':
        pipeline_id = '74f5ccb1-053a-46cf-ad7f-005f67a15652'
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))

    d3m_index.register_primitive(
        RandomSamplingImputer.metadata.query()['python_path'],
        RandomSamplingImputer
    )

    pipeline = pipeline_module.Pipeline(pipeline_id)
    pipeline.add_input(name='inputs')
    step_counter = 0


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.dataset_to_dataframe.Common'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='inputs.0'
    )
    step.add_output('produce')
    pipeline.add_step(step)
    raw_data_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.column_parser.DataFrameCommon'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    parsed_data_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'
        )
    )
    step.add_hyperparameter(
        name='semantic_types', argument_type=metadata_base.ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/Attribute']
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=parsed_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    raw_attributes_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'
        )
    )
    step.add_hyperparameter(
        name='semantic_types', argument_type=metadata_base.ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=parsed_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    true_targets_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_preprocessing.random_sampling_imputer.BYU'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_attributes_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    imputed_attributes_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    if task_type == 'regression':
        step = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                'd3m.primitives.regression.random_forest.SKlearn'
            )
        )
        step.add_hyperparameter(
            name='use_semantic_types',
            argument_type=metadata_base.ArgumentType.VALUE, data=True
        )
        step.add_argument(
            name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=imputed_attributes_data_reference
        )
        step.add_argument(
            name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=true_targets_data_reference
        )
        step.add_output('produce')
        pipeline.add_step(step)
        step_counter += 1


    elif task_type == 'classification':
        step = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                'd3m.primitives.classification.random_forest.SKlearn'
            )
        )
        step.add_hyperparameter(
            name='use_semantic_types',
            argument_type=metadata_base.ArgumentType.VALUE, data=True
        )
        step.add_argument(
            name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=imputed_attributes_data_reference
        )
        step.add_argument(
            name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=true_targets_data_reference
        )
        step.add_output('produce')
        pipeline.add_step(step)
        step_counter += 1

    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='steps.{}.produce'.format(step_counter - 1)
    )
    step.add_argument(
        name='reference', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    step_counter += 1


    pipeline.add_output(
        name='predictions',
        data_reference='steps.{}.produce'.format(step_counter - 1)
    )

    return pipeline


def generate_metafeature_pipeline(task_type):
    if task_type == 'classification':
        pipeline_id = 'b32b9af1-34b4-437b-ad83-650f7df10acb'
    elif task_type == 'regression':
        pipeline_id = '3013ad40-7c51-4991-b0fb-dbec65607979'
    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))

    d3m_index.register_primitive(
        MetafeatureExtractor.metadata.query()['python_path'],
        MetafeatureExtractor
    )

    pipeline = pipeline_module.Pipeline(pipeline_id)
    pipeline.add_input(name='inputs')
    step_counter = 0


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.dataset_to_dataframe.Common'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='inputs.0'
    )
    step.add_output('produce')
    pipeline.add_step(step)
    raw_data_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.column_parser.DataFrameCommon'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    parsed_data_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.metalearning.metafeature_extractor.BYU'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=parsed_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    imputed_data_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'
        )
    )
    step.add_hyperparameter(
        name='semantic_types', argument_type=metadata_base.ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/Attribute']
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=parsed_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    raw_attributes_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'
        )
    )
    step.add_hyperparameter(
        name='semantic_types', argument_type=metadata_base.ArgumentType.VALUE,
        data=['https://metadata.datadrivendiscovery.org/types/TrueTarget']
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=parsed_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    true_targets_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_cleaning.imputer.SKlearn'
        )
    )
    step.add_hyperparameter(
        name='use_semantic_types',
        argument_type=metadata_base.ArgumentType.VALUE, data=True
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_attributes_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    imputed_attributes_data_reference = 'steps.{}.produce'.format(step_counter)
    step_counter += 1


    if task_type == 'regression':
        step = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                'd3m.primitives.regression.random_forest.SKlearn'
            )
        )
        step.add_hyperparameter(
            name='use_semantic_types',
            argument_type=metadata_base.ArgumentType.VALUE, data=True
        )
        step.add_argument(
            name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=imputed_attributes_data_reference
        )
        step.add_argument(
            name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=true_targets_data_reference
        )
        step.add_output('produce')
        pipeline.add_step(step)
        step_counter += 1


    elif task_type == 'classification':
        step = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                'd3m.primitives.classification.random_forest.SKlearn'
            )
        )
        step.add_hyperparameter(
            name='use_semantic_types',
            argument_type=metadata_base.ArgumentType.VALUE, data=True
        )
        step.add_argument(
            name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=imputed_attributes_data_reference
        )
        step.add_argument(
            name='outputs', argument_type=metadata_base.ArgumentType.CONTAINER,
            data_reference=true_targets_data_reference
        )
        step.add_output('produce')
        pipeline.add_step(step)
        step_counter += 1

    else:
        raise ValueError('Invalid task_type: {}'.format(task_type))


    step = pipeline_module.PrimitiveStep(
        primitive=d3m_index.get_primitive(
            'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'
        )
    )
    step.add_argument(
        name='inputs', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference='steps.{}.produce'.format(step_counter - 1)
    )
    step.add_argument(
        name='reference', argument_type=metadata_base.ArgumentType.CONTAINER,
        data_reference=raw_data_data_reference
    )
    step.add_output('produce')
    pipeline.add_step(step)
    step_counter += 1


    pipeline.add_output(
        name='predictions',
        data_reference='steps.{}.produce'.format(step_counter - 1)
    )

    return pipeline

def remove_digests(
    pipeline_json_structure, *, exclude_primitives: set = set()
):
    """
    Parameters
    ----------
    exclude_primitives: set
        removes digest from all primitive steps except for those whose id is in
        exclude_primitives
    """
    if 'digest' in pipeline_json_structure:
        del pipeline_json_structure['digest']

    for step in pipeline_json_structure['steps']:
        if step['primitive']['id'] not in exclude_primitives:
            del step['primitive']['digest']
            if step["primitive"]["id"] == "dsbox-ensemble-voting":
                step["primitive"]["id"]
        

    return pipeline_json_structure


def update_digest(
    pipeline_json_structure, filename=None
):
    """
    This function updates the pipeline's digests and version numbers

    Parameters
    ----------
    pipeline_json_structure: the pipeline in JSON form (WITHOUT) digests.  This or the `filename` parameter is mandatory
    filename: the filename of the pipeline json, so we can read it in

    :return a pipeline with updated digests
    """
    if pipeline_json_structure is None and filename is None:
        raise Exception
    elif pipeline_json_structure is None:
        with open(filename, "r") as file:
            # NOTE: must be a pipeline with no digests, or recent digests
            # NOTE: reading this in as straight JSON doesn't work so we have to use the pipeline_module
            pipeline_json_structure = pipeline_module.Pipeline.from_json(string_or_file=file).to_json_structure()
    else:
        try:
            pipeline_json_structure = pipeline_module.Pipeline.from_json(json.dumps(pipeline_json_structure)).to_json_structure()
        except Exception as e:
            pass
    for step in pipeline_json_structure['steps']:
        # if not updated, check and update
        primitive = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                step["primitive"]["python_path"]
            )
        )
        check_step = primitive.to_json_structure()

        # lets verify that both are updated
        id_matches = check_step["primitive"]["id"] == step["primitive"]["id"]
        if not id_matches:
            step["primitive"]["id"] = check_step["primitive"]["id"]
        version_matches = check_step["primitive"]["version"] == step["primitive"]["version"]
        if not version_matches:
            step["primitive"]["version"] = check_step["primitive"]["version"]
        # make sure the digest exists
        digests_match = "digest" in step["primitive"] and check_step["primitive"]["digest"] == step["primitive"]["digest"]
        if not digests_match:
            step["primitive"]["digest"] = check_step["primitive"]["digest"]

    return pipeline_json_structure

def get_pipeline_from_database(pipeline_id, mongo_client):
    """
    This function gets a pipeline from our local database given an id

    Parameters
    ----------
    pipeline_id: the id of the pipeline to grab
    mongo_client: a connection to the database

    :return a pipeline matching the id
    """
    collection = mongo_client.metalearning.pipelines
    pipeline_to_write = collection.find({"id": pipeline_id})
    for pipeline in pipeline_to_write:
        # should only be one pipeline
        return pipeline

def add_best_pipelines():
    """
    This function checks the best_pipelines.csv for the best pipelines for a dataset, prepares and updates it, and writes it to the submodule.
    It also check how many pipelines beat MIT-LL and the EXlines.
    """
    mongo_client = pymongo.MongoClient(lab_hostname, real_mongo_port)

    imputer_version = None
    best_pipelines_df = pd.read_csv("submission/pipelines/best_pipelines.csv", index_col=0)

    beat_mit = 0
    beat_exlines = 0
    has_pipeline = 0
    for index, dataset in enumerate(best_pipelines_df):
        dataset_id = dataset.replace("_dataset", "", 1)
        if dataset_id not in list(seed_datasets_exlines.keys()):
            continue

        # grab the best pipeline
        pipelines = best_pipelines_df[dataset]
        best_pipeline_id = pipelines.idxmax()
        best_pipeline_score = pipelines.max()
        has_pipeline += 1

        # See how well we do compared to others
        problem_details = seed_datasets_exlines[dataset_id]
        if problem_details["problem"] == "accuracy":
            if problem_details["score"] <= best_pipeline_score:
                beat_exlines += 1
            if problem_details["mit-score"] <= best_pipeline_score:
                beat_mit += 1
        else:
            ## is regression
            if problem_details["score"] >= best_pipeline_score:
                beat_exlines += 1
            if problem_details["mit-score"] >= best_pipeline_score:
                beat_mit += 1

        # get the best pipeline and update it
        best_pipeline_json = get_pipeline_from_database(best_pipeline_id, mongo_client)
        del best_pipeline_json["_id"]
        no_digest_pipeline = remove_digests(best_pipeline_json)
        updated_pipeline = update_digest(no_digest_pipeline)
            
        # get directory to put new pipelines
        if imputer_version == None:
            _, imputer_version = extract_byu_info(updated_pipeline)
            IMPUTER_PIPELINE_PATH = os.path.join(byu_dir, __imputer_path__, imputer_version, "pipelines/")

        # prepare meta file
        seed = dataset_id in list(seed_datasets_exlines.keys())
        meta_file = create_meta_script_seed(dataset, seed=seed)

        print("Writing pipeline for dataset: {} to {}".format(dataset, IMPUTER_PIPELINE_PATH + best_pipeline_id + ".json"))
        with open(IMPUTER_PIPELINE_PATH + best_pipeline_id + ".json", "w") as file:
            file.write(json.dumps(updated_pipeline, indent=4))

        with open(IMPUTER_PIPELINE_PATH + best_pipeline_id + ".meta", "w") as file:
            file.write(json.dumps(meta_file, indent=4))


    print("############## RESULTS #################")
    print(beat_mit, " pipelines beat MIT")
    print(beat_exlines, " pipelines beat EXlines")
    print(has_pipeline, " pipelines for seed datasets")


if __name__ == "__main__":
    # get directory ready
    byu_dir = get_new_d3m_path()
    clear_directory(byu_dir)

    for task_type in ['classification', 'regression']:
        # generate and update imputer
        pipeline = generate_imputer_pipeline(task_type)
        pipeline_json_structure = pipeline.to_json_structure()
        pipeline_json_structure = remove_digests(pipeline_json_structure, exclude_primitives={
                                                 RandomSamplingImputer.metadata.query()['id']})
        pipeline_json_structure = update_digest(pipeline_json_structure)
        # place in submodule
        imputer_path, imputer_version = extract_byu_info(pipeline_json_structure)
        os.environ['imputer_location'] = os.path.join(byu_dir, imputer_path)
        create_and_add_to_directory(os.path.join(byu_dir, imputer_path), str(imputer_version), pipeline_json_structure)

        # generate and update metafeatures
        pipeline = generate_metafeature_pipeline(task_type)
        pipeline_json_structure = pipeline.to_json_structure()
        pipeline_json_structure = remove_digests(pipeline_json_structure, exclude_primitives={
                                                 MetafeatureExtractor.metadata.query()['id']})
        pipeline_json_structure = update_digest(pipeline_json_structure)
        # place in submodule
        metafeature_path, metafeatures_version = extract_byu_info(pipeline_json_structure)
        os.environ['metafeature_location'] = os.path.join(byu_dir, metafeature_path)
        create_and_add_to_directory(os.path.join(byu_dir, metafeature_path), str(metafeatures_version), pipeline_json_structure)

        # add any other pipelines
        add_best_pipelines()

