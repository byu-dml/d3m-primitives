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

if __name__ == '__main__':
    dataframe = load_dataset()
    # dataframe['Class'] = np.NaN
    target_col_metadata = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS,30)))
    # target_col_semantic_types = target_col_metadata.get('semantic_types', []) + ('https://metadata.datadrivendiscovery.org/types/RedactedTarget',)
    target_col_semantic_types = target_col_metadata.get('semantic_types', []) + ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)
    # target_col_semantic_types = target_col_metadata.get('semantic_types', [])
    target_col_metadata['semantic_types'] = target_col_semantic_types
    dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS,30), target_col_metadata)
    hyperparams_class = MetafeatureExtractor.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    hyperparams = hyperparams_class(metafeature_subset='INEXPENSIVE', metafeatures_to_compute=['linear_discriminant_analysis'])
    df_with_metafeatures = MetafeatureExtractor(hyperparams=hyperparams).produce(inputs=dataframe).value
    md = df_with_metafeatures.metadata.to_json_structure()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(md[0]['metadata']['data_metafeatures'])
    print('done testing d3m-loaded dataset')

    # FOR TESTING WITH JUST READ_CSV
    # dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/38_sick_dataset/tables/learningData.csv')
    # dataframe = DataFrame(pd.read_csv(dataset_path))
    # target_col_metadata = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS,30)))
    # target_col_semantic_types = target_col_metadata.get('semantic_types', ()) + ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)
    # target_col_metadata['semantic_types'] = target_col_semantic_types
    # dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS,30), target_col_metadata)
    # hyperparams_class = MetafeatureExtractor.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    # df_with_metafeatures = MetafeatureExtractor(hyperparams=hyperparams_class(metafeature_subset='ALL', metafeatures_to_compute=['class_probabilities', 'means_of_attributes'])).produce(inputs=dataframe).value
    # df_with_metafeatures.metadata.pretty_print(())
    # print('done testing pandas-loaded dataset')