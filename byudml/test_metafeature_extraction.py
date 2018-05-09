import pandas as pd
import os
import json

from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.metadata import base as metadata_base
from load_d3m_dataset import load_dataset

from metafeature_extraction import MetafeatureExtractor

if __name__ == '__main__':
    dataframe = load_dataset()

    target_col_metadata = dict(dataframe.metadata.query((metadata_base.ALL_ELEMENTS,30)))
    target_col_semantic_types = target_col_metadata.get('semantic_types', []) + ('https://metadata.datadrivendiscovery.org/types/Target',)
    target_col_metadata['semantic_types'] = target_col_semantic_types
    dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS,30), target_col_metadata)

    metafeatures = MetafeatureExtractor(hyperparams=None).produce(inputs=dataframe).value
    print(metafeatures)