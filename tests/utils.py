from typing import List, NamedTuple
import argparse
import inspect
import json
import os

from d3m import cli
from d3m import container
from d3m.metadata import problem as problem_module
from d3m.metadata import base as metadata_base
import pandas as pd

DEFAULT_DATASET_DIR = '/datasets/training_datasets/LL0'
DATASETS_DIR = '/datasets/seed_datasets_current'

# Semantic Types
ATTRIBUTE = 'https://metadata.datadrivendiscovery.org/types/Attribute'
PREDICTED_TARGET = 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'

def get_dataset_doc_path(dataset_name, dataset_dir=DEFAULT_DATASET_DIR):
    return os.path.join(
        dataset_dir, dataset_name, dataset_name + '_dataset', 'datasetDoc.json'
    )


def get_dataset_doc(dataset_name, dataset_dir=DEFAULT_DATASET_DIR):
    dataset_doc_path = get_dataset_doc_path(dataset_name, dataset_dir)
    with open(dataset_doc_path, 'r') as f:
        dataset_doc = json.load(f)
    return dataset_doc


def get_problem_path(dataset_name, dataset_dir=DEFAULT_DATASET_DIR):
    return os.path.join(
        dataset_dir, dataset_name, dataset_name + '_problem', 'problemDoc.json'
    )


def get_problem(problem_path, *, parse=True):
    if parse:
        problem_description = problem_module.parse_problem_description(problem_path)
    else:
        with open(problem_path, 'r') as f:
            problem_description = json.load(f)
    return problem_description


def get_default_args(f):
    return {
        k: v.default for k, v in inspect.signature(f).parameters.items()
        # if v.default is not inspect.Parameter.empty
    }


def get_data_splits_path(dataset_name, dataset_dir=DEFAULT_DATASET_DIR):
    return os.path.join(
        dataset_dir, dataset_name, dataset_name + '_problem', 'dataSplits.csv'
    )

class Column(NamedTuple):
    """
    Represents a column in a d3m.container.DataFrame.
    Used to make constructing d3m dataframes used for
    testing more simple.
    """
    name: str
    data: list
    semantic_types: List[str]

def make_df(cols: List[Column]) -> container.DataFrame:
    """
    Builds a d3m.container.DataFrame. Used to make constructing
    d3m dataframes used for testing more simple.
    """
    # Make the data
    df = container.DataFrame(
        pd.DataFrame({ col.name: col.data for col in cols }),
        generate_metadata=True
    )
    # Add each column's semantic types
    for col in cols:
        for semantic_type in col.semantic_types:
            df.metadata = df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, df.columns.get_loc(col.name)),
                semantic_type
            )
    return df

def print_df(df: container.DataFrame, name: str) -> None:
    """Debug helper"""
    print(f'\n"{name}" Data Frame:')
    print(df)
    print('column dtypes:')
    print(df.dtypes)
    print('column metadata:')
    for col_i in range(df.shape[1]):
        print(df.metadata.query_column(col_i))

def are_df_elements_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Checks if all elemenst are equal between the dataframes using the
    `==` operator. the pandas.DataFrame.equals method is finicky and
    won't always return true when you expect it too.
    """
    return (df1 == df2).all(axis=None)


class D3MDatasetUtil:

    def __init__(
        self, dataset_dir: str = DATASETS_DIR,
        dataset_name: str = '185_baseball'
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name

        self.dataset_doc_path = get_dataset_doc_path(
            self.dataset_name, self.dataset_dir
        )
        self.problem_path = get_problem_path(
            self.dataset_name, self.dataset_dir
        )
        self.data_splits_path = get_data_splits_path(
            self.dataset_name, self.dataset_dir
        )

def evaluate_pipeline(
    pipeline_path, data_pipeline_path, data_splits_path, scoring_pipeline_path,
    dataset_doc_path, problem_path
):
    parser = argparse.ArgumentParser(description='Run D3M pipelines with default hyper-parameters.')
    cli.runtime_configure_parser(parser)
    test_args = [
        'evaluate',
        '-p', pipeline_path,
        '-d', data_pipeline_path,
        '--data-split-file', data_splits_path,
        '-n', scoring_pipeline_path,
        '-r', problem_path,
        '-i', dataset_doc_path,
    ]

    arguments = parser.parse_args(args=test_args)

    cli.runtime_handler(arguments, parser)
