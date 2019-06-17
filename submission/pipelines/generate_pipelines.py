import json
import os
import pymongo
import pandas as pd
from d3m import index as d3m_index
from d3m.metadata import (
    base as metadata_base, pipeline as pipeline_module
)

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor
from byudml import __imputer_version__, __imputer_path__,  __metafeature_version__,  __metafeature_path__
import sys
sys.path.append('.')
from submission.utils import (get_new_d3m_path, clear_directory, write_pipeline_for_submission, get_pipeline_from_database, \
                              seed_datasets_exlines, create_meta_script_seed)

real_mongo_port = 12345
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

    # delete any extra inputs.  TODO: change the experimenter to not do this
    while len(pipeline_json_structure["inputs"]) > 1:
        del pipeline_json_structure["inputs"][-1]
        
    return pipeline_json_structure


def update_pipeline(
    pipeline_to_update, filename=None
):
    """
    This function updates the pipeline's digests and version numbers

    Parameters
    ----------
    pipeline_json_structure: the pipeline in JSON form (WITHOUT) digests.  This or the `filename` parameter is mandatory
    filename: the filename of the pipeline json, so we can read it in

    :return a pipeline with updated digests
    """
    if pipeline_to_update is None and filename is None:
        raise ValueError("No pipeline json was given")
    elif pipeline_to_update is None:
        with open(filename, "r") as file:
            # NOTE: must be a pipeline with no digests, or recent digests
            # NOTE: reading this in as straight JSON doesn't work so we have to use the pipeline_module
            pipeline_to_update = pipeline_module.Pipeline.from_json(string_or_file=file).to_json_structure()
    else:
        try:
            pipeline_to_update = pipeline_module.Pipeline.from_json(json.dumps(pipeline_to_update)).to_json_structure()
        except Exception as e:
            pass
    for step in pipeline_to_update['steps']:
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

    return pipeline_to_update
        
def add_best_pipelines(base_dir):
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
        updated_pipeline = update_pipeline(no_digest_pipeline)
            
        # get directory to put new pipelines
        if imputer_version == None:
            IMPUTER_PIPELINE_PATH = os.path.join(base_dir, __imputer_path__, __imputer_version__, "pipelines/")

        # prepare meta file
        seed = dataset_id in list(seed_datasets_exlines.keys())
        meta_file = create_meta_script_seed(dataset_id, seed=seed)

        print("Writing pipeline for dataset: {} to {}".format(dataset, IMPUTER_PIPELINE_PATH + best_pipeline_id + ".json"))
        with open(IMPUTER_PIPELINE_PATH + best_pipeline_id + ".json", "w") as file:
            file.write(json.dumps(updated_pipeline, indent=4))

        with open(IMPUTER_PIPELINE_PATH + best_pipeline_id + ".meta", "w") as file:
            file.write(json.dumps(meta_file, indent=4))


    print("############## RESULTS #################")
    print(beat_mit, " pipelines beat MIT")
    print(beat_exlines, " pipelines beat EXlines")
    print(has_pipeline, " pipelines for seed datasets")

def main():
    # get directory ready
    byu_dir = get_new_d3m_path()
    clear_directory(byu_dir)

    # add our basic pipelines
    for (problem_type, problem_name) in [('classification', '185_baseball'), ('regression', '196_autoMpg')]:
        # generate and update imputer
        pipeline = generate_imputer_pipeline(problem_type)
        pipeline_json_structure = pipeline.to_json_structure()
        pipeline_json_structure = remove_digests(pipeline_json_structure, exclude_primitives={
                                                 RandomSamplingImputer.metadata.query()['id']})
        pipeline_json_structure = update_pipeline(pipeline_json_structure)
        # place in submodule
        write_pipeline_for_submission(os.path.join(byu_dir, __imputer_path__), str(__imputer_version__), pipeline_json_structure, problem_name)

        # generate and update metafeatures
        pipeline = generate_metafeature_pipeline(problem_type)
        pipeline_json_structure = pipeline.to_json_structure()
        pipeline_json_structure = remove_digests(pipeline_json_structure, exclude_primitives={
                                                 MetafeatureExtractor.metadata.query()['id']})
        pipeline_json_structure = update_pipeline(pipeline_json_structure)
        # place in submodule
        write_pipeline_for_submission(os.path.join(byu_dir, __metafeature_path__), str(__metafeature_version__), pipeline_json_structure, problem_name)

    # add other best pipelines
    add_best_pipelines(byu_dir)

if __name__ == '__main__':
    main()
