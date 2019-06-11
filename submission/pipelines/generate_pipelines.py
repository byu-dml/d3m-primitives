import json
import os

from d3m import index as d3m_index
from d3m.metadata import (
    base as metadata_base, pipeline as pipeline_module
)

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor
from byudml import __imputer_version__, __imputer_path__,  __metafeature_version__,  __metafeature_path__
import sys
sys.path.append('./submission')
from utils import get_new_d3m_path, clear_directory, write_pipeline_for_submission


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

    return pipeline_json_structure


def update_pipeline(
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
        raise ValueError("No pipeline json was given")
    elif pipeline_json_structure is None:
        with open(filename, "r") as file:
            # NOTE: must be a pipeline with no digests, or recent digests
            # NOTE: reading this in as straight JSON doesn't work so we have to use the pipeline_module
            pipeline_to_update = pipeline_module.Pipeline.from_json(string_or_file=file).to_json_structure()
    else:
        pipeline_to_update = pipeline_module.Pipeline.from_json(json.dumps(pipeline_json_structure)).to_json_structure()

    for step in pipeline_to_update['steps']:
        # if not updated, check and update
        primitive = pipeline_module.PrimitiveStep(
            primitive=d3m_index.get_primitive(
                step["primitive"]["python_path"]
            )
        )
        check_step = primitive.to_json_structure()
        # lets verify that both are updated
        assert check_step["primitive"]["version"] == step["primitive"]["version"], "Updating version failed"

    return pipeline_to_update


def main():
    # get directory ready
    byu_dir = get_new_d3m_path()
    clear_directory(byu_dir)

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

if __name__ == '__main__':
    main()
