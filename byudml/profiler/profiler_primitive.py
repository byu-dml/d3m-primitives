import collections
import copy
import os.path
import re
import typing
import multiprocessing as mp
import pickle
import sys
import zipfile

import numpy as np # type: ignore
import pandas as pd # type: ignore
from pandas.io import parsers as pandas_parsers  # type: ignore
from sentence_transformers import SentenceTransformer

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, params
from d3m.primitive_interfaces import base, unsupervised_learning

from byudml import __version__ as __package_version__
from byudml import __profiler_path__, __profiler_version__

from common_primitives.simple_profiler import (
    Inputs, Outputs, Params, Hyperparams, SimpleProfilerPrimitive
)

__all__ = ('SemanticProfilerPrimitive',)


class SemanticProfilerPrimitive(SimpleProfilerPrimitive):
    """
    A primitive that determines missing semantic types for columns and adds
    them automatically. This subclasses the D3M Common Primitives' SimpleProfilerPrimitive and overrides some that primitive's core profiling functionality. This primitive uses a metamodel trained on the D3M column type annotations to predict some column types.
    """

    __author__ = 'Brandon Schoenfeld'
    _weights_configs = [
        {
            'type': 'FILE',
            'key': 'distilbert-base-nli-stsb-mean-tokens.zip',
            'file_uri': 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distilbert-base-nli-stsb-mean-tokens.zip',
            'file_digest': '87a361052ca09566b805bdfa168dc775eaa689686a1124401868a9b96fcbc19f',
        },
    ]
    metadata = metadata_base.PrimitiveMetadata({
        'id': 'af214333-e67b-4e59-a49b-b16f5501a925',
        'version': __profiler_version__,
        'name': 'Semantic Profiler',
        'description': 'This primitive is an adapatation of the d3m common profiler (https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/c170029e9a0f875af28c6b9af20adc90bd4df0bb/common_primitives/simple_profiler.py). It predicts semantic column types using a natural language embeddings of the the column name. The internal model uses these embeddings to predict the semantic types found in the dataset annotations created by MIT Lincoln Labs.',
        'python_path': __profiler_path__,
        'source': {
            'name': 'byu-dml',
            'contact': 'mailto:bjschoenfeld@gmail.com',
            'uris': [
                'https://github.com/byu-dml/d3m-primitives'
            ],
        },
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'byudml',
                'version': __package_version__
            },
        ] + _weights_configs,
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.SCHEMA_DISCOVERY,
    })

    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Optional[typing.Dict[str, str]]=None) -> None:
        super().__init__(hyperparams=hyperparams, volumes=volumes)

        self._emb_model = self._init_embedding_model()
        self._profiler_model = self._init_profiler_model()

    def _init_embedding_model(self) -> SentenceTransformer:
        weights_path = self._find_weights_path(self._weights_configs[0]['key'])
        weights_path = self._extract_weights(weights_path)
        with d3m_utils.silence():
            emb_model = SentenceTransformer(weights_path)
        return emb_model

    def _find_weights_path(self, key_filename):
        if key_filename in self.volumes:
            weight_file_path = self.volumes[key_filename]
        else:
            weight_file_path = os.path.join('.', self._weights_configs['file_digest'], key_filename)

        if not os.path.isfile(weight_file_path):
            raise ValueError(
                "Can't get weights file from volumes by key '{key_filename}' and at path '{path}'.".format(
                    key_filename=key_filename,
                    path=weight_file_path,
                ),
            )

        return weight_file_path

    def _extract_weights(self, weights_path):
        extracted_weights_path = weights_path[:-4]  # remove .zip

        if not os.path.isfile(extracted_weights_path):
            with zipfile.ZipFile(weights_path, 'r') as zf:
                zf.extractall(extracted_weights_path)

        return extracted_weights_path

    def _init_profiler_model(self):
        zip_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.zip')
        model_path = os.path.join(self._extract_weights(zip_model_path), 'model.bin')

        with open(model_path, 'rb') as f:
            profiler_model = pickle.load(f)
        return profiler_model

    def _predict_semantic_type(self, input_column: container.DataFrame) -> str:
        column_name = input_column.metadata.query(('ALL_ELEMENTS', 0))['name']

        with d3m_utils.silence():
            column_name_emb = self._emb_model.encode([column_name.lower()], show_progress_bar=False)

        prediction = self._profiler_model.predict(column_name_emb)
        assert prediction.shape[0] == 1

        return prediction[0]

    def _is_boolean(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'boolean'

    def _is_categorical(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'categorical'

    def _is_integer(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'integer'

    def _is_text(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'string'

    def _is_datetime(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'dateTime'

    def _is_float(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'real'

    def _is_float_vector(self, input_column: container.DataFrame) -> bool:
        return self._predict_semantic_type(input_column) == 'realVector'

