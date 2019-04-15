import pandas as pd
import os
import json
import numpy as np
import pprint

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.metadata import base as metadata_base
from load_d3m_dataset import load_dataset

from d3m.primitives.metafeature_extraction.metafeature_extractor import BYU as MetafeatureExtractor


def mark_targets(dataframe):
    for col_pos in range(len(dataframe.columns)):
        column_metadata = dataframe.metadata.query_column(col_pos)
        semantic_types = column_metadata.get('semantic_types', tuple())
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
            dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, col_pos), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')

def test_metafeature_extraction(dataframe):
    mark_targets(dataframe)
    hyperparams_class = MetafeatureExtractor.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    hyperparams = hyperparams_class(metafeature_subset='INEXPENSIVE', metafeatures_to_compute=['linear_discriminant_analysis'])
    df_with_metafeatures = MetafeatureExtractor(hyperparams=hyperparams).produce(inputs=dataframe).value
    md = df_with_metafeatures.metadata.to_json_structure()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(md[0]['metadata']['data_metafeatures'])


if __name__ == '__main__':
    
    for dataset_name in ('38_sick', '196_autoMpg'):
        dataframe = load_dataset(dataset_name)
        test_metafeature_extraction(dataframe)