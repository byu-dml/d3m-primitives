"""
This code has been copied and modified from https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/c170029e9a0f875af28c6b9af20adc90bd4df0bb/common_primitives/simple_profiler.py
"""

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

import common_primitives
from common_primitives import utils

__all__ = ('SimpleProfilerPrimitive',)

WHITESPACE_REGEX = re.compile(r'\s')

if hasattr(pandas_parsers, 'STR_NA_VALUES'):
    NA_VALUES = pandas_parsers.STR_NA_VALUES
else:
    # Backwards compatibility for Pandas before 1.0.0.
    NA_VALUES = pandas_parsers._NA_VALUES

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    add_semantic_types: typing.Optional[typing.List[typing.List[str]]]
    remove_semantic_types: typing.Optional[typing.List[typing.List[str]]]


class Hyperparams(hyperparams_module.Hyperparams):
    detect_semantic_types = hyperparams_module.Set(
        elements=hyperparams_module.Enumeration(
            values=[
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
                'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
                'https://metadata.datadrivendiscovery.org/types/UniqueKey',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/Time',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/UnknownType',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
            ],
            default='http://schema.org/Boolean',
        ),
        default=(
            'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
            'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
            'https://metadata.datadrivendiscovery.org/types/UniqueKey',
            'https://metadata.datadrivendiscovery.org/types/Attribute',
            'https://metadata.datadrivendiscovery.org/types/Time',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            'https://metadata.datadrivendiscovery.org/types/UnknownType',
            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of semantic types to detect and set. One can provide a subset of supported semantic types to limit what the primitive detects.",
    )
    remove_unknown_type = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Remove \"https://metadata.datadrivendiscovery.org/types/UnknownType\" semantic type from columns on which the primitive has detected other semantic types.",
    )
    categorical_max_absolute_distinct_values = hyperparams_module.Union[typing.Union[int, None]](
        configuration=collections.OrderedDict(
            limit=hyperparams_module.Bounded[int](
                lower=1,
                upper=None,
                default=50,
            ),
            unlimited=hyperparams_module.Hyperparameter[None](
                default=None,
                description='No absolute limit on distinct values.',
            ),
        ),
        default='limit',
        description='The maximum absolute number of distinct values (all missing values as counted as one distinct value) for a column to be considered categorical.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    categorical_max_ratio_distinct_values = hyperparams_module.Bounded[float](
        lower=0,
        upper=1,
        default=0.05,
        description='The maximum ratio of distinct values (all missing values as counted as one distinct value) vs. number of rows for a column to be considered categorical.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    nan_values = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[str](''),
        default=sorted(NA_VALUES),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of strings to recognize as NaNs when detecting a float column.",
    )
    text_min_ratio_values_with_whitespace = hyperparams_module.Bounded[float](
        lower=0,
        upper=1,
        default=0.5,
        description='The minimum ratio of values with any whitespace (after first stripping) vs. number of rows for a column to be considered a text column.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    use_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be detected, it is skipped.",
    )
    exclude_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams_module.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should detected columns be appended, should they replace original columns, or should only detected columns be returned?",
    )
    add_index_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    replace_index_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Replace primary index columns even if otherwise appending columns. Applicable only if \"return_result\" is set to \"append\".",
    )


class SemanticProfilerPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which determines missing semantic types for columns and adds
    them automatically. It uses a set of hard-coded rules/heuristics to determine
    semantic types. Feel free to propose improvements.

    Besides determining column types it also determines some column roles.

    Some rules are intuitive and expected, but there are also few special behaviors
    (if not disabled by not providing a corresponding semantic type in
    ``detect_semantic_types``):

    * If a column does not have any semantic types,
      ``https://metadata.datadrivendiscovery.org/types/UnknownType`` semantic type
      is first set for the column. If any other semantic type is set later on as
      part of logic of this primitive, the
      ``https://metadata.datadrivendiscovery.org/types/UnknownType`` is removed
      (including if the column originally came with this semantic type).
    * If a column has ``https://metadata.datadrivendiscovery.org/types/SuggestedTarget``
      semantic type and no other column (even those not otherwise operated on by
      the primitive) has a semantic type
      ``https://metadata.datadrivendiscovery.org/types/TrueTarget`` is set on
      the column. This allows operation on data without a problem description.
      This is only for the first such column.
    * All other columns which are missing semantic types initially we set as
      ``https://metadata.datadrivendiscovery.org/types/Attribute``.
    * Any column with ``http://schema.org/DateTime`` semantic type is also set
      as ``https://metadata.datadrivendiscovery.org/types/Time`` semantic type.
    * ``https://metadata.datadrivendiscovery.org/types/PrimaryKey`` or
      ``https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey`` is set only
      if no other column (even those not otherwise operated on by
      the primitive) is a primary key, and set based on the column name: only
      when it is ``d3mIndex``.
    """

    __author__ = 'Brandon Schoenfeld'
    _weights_configs = [
        {
            'type': 'FILE',
            'key': 'distilbert-base-nli-stsb-mean-tokens.zip',
            'file_uri': 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distilbert-base-nli-stsb-mean-tokens.zip',
            'file_digest': '9d3dfdf353743741a4be36250868636d377b0dcc16067bfcdfc192fe7239d7c8',
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

        self._training_inputs: Inputs = None
        self._add_semantic_types: typing.List[typing.List[str]] = None
        self._remove_semantic_types: typing.List[typing.List[str]] = None
        self._fitted: bool = False

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

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

        self._dataset_name = ''  # todo
        self._dataset_description = ''  # todo

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        # The logic of detecting values tries to mirror also the logic of parsing
        # values in "ColumnParserPrimitive". One should keep them in sync.

        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        self._add_semantic_types, self._remove_semantic_types = self._fit_columns(self._training_inputs)
        self._fitted = True

        return base.CallResult(None)

    def _fit_columns(self, inputs: Inputs) -> typing.Tuple[typing.List[typing.List[str]], typing.List[typing.List[str]]]:
        true_target_columns = inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        index_columns = inputs.metadata.get_index_columns()

        # Target and index columns should be set only once, if they are set.
        has_set_target_columns = False
        has_set_index_column = False

        columns_to_use = self._get_columns(inputs.metadata)

        fitted_add_semantic_types = []
        fitted_remove_semantic_types = []

        for column_index in columns_to_use:
            input_column = inputs.select_columns([column_index])
            column_metadata = inputs.metadata.query_column(column_index)
            column_name = column_metadata.get('name', str(column_index))
            column_semantic_types = list(column_metadata.get('semantic_types', []))

            # We might be here because column has a known type, but it has "https://metadata.datadrivendiscovery.org/types/SuggestedTarget" set.
            has_unknown_type = not column_semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in column_semantic_types

            # A normalized copy of semantic types, which always includes unknown type.
            normalized_column_semantic_types = copy.copy(column_semantic_types)

            # If we are processing this column and it does not have semantic type that it has missing semantic types,
            # we first set it, to normalize the input semantic types. If we will add any other semantic type,
            # we will then remove this semantic type.
            if has_unknown_type \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in self.hyperparams['detect_semantic_types'] \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' not in normalized_column_semantic_types:
                normalized_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/UnknownType')

            # A working copy of semantic types.
            new_column_semantic_types = copy.copy(normalized_column_semantic_types)

            if has_unknown_type:
                is_float = self._is_float(input_column)
                is_integer = self._is_integer(input_column)

                # If it looks like proper float (so not integer encoded as float), then we do not detect it as boolean.
                if self._is_boolean(input_column) \
                        and (not is_float or is_integer) \
                        and 'http://schema.org/Boolean' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Boolean' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Boolean')

                # If it looks like proper float (so not integer encoded as float), then we do not detect it as categorical.
                elif self._is_categorical(input_column) \
                        and (not is_float or is_integer) \
                        and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/CategoricalData')

                elif is_integer \
                        and 'http://schema.org/Integer' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Integer' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Integer')

                elif is_float \
                        and 'http://schema.org/Float' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Float' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Float')

                elif self._is_float_vector(input_column) \
                        and 'https://metadata.datadrivendiscovery.org/types/FloatVector' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/FloatVector' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/FloatVector')

                elif self._is_datetime(input_column) \
                        and 'http://schema.org/DateTime' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/DateTime' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/DateTime')

                elif self._is_text(input_column) \
                        and 'http://schema.org/Text' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/Text' not in new_column_semantic_types:
                    new_column_semantic_types.append('http://schema.org/Text')

                if 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in self.hyperparams['detect_semantic_types'] \
                        and self._is_unique_key(input_column) \
                        and 'http://schema.org/Text' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/UniqueKey' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/UniqueKey')

            if not true_target_columns \
                    and not has_set_target_columns \
                    and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in self.hyperparams['detect_semantic_types'] \
                    and 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in new_column_semantic_types:
                # It should not be set because there are no columns with this semantic type in whole DataFrame.
                assert 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types
                new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                if 'https://metadata.datadrivendiscovery.org/types/Target' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                    new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                has_set_target_columns = True

            if has_unknown_type:
                if not index_columns and not has_set_index_column:
                    if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in self.hyperparams['detect_semantic_types'] \
                            and column_name == 'd3mIndex' \
                            and 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in new_column_semantic_types:
                        # It should not be set because there are no columns with this semantic type in whole DataFrame.
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
                        new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
                        new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UniqueKey')
                        if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                            new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                        has_set_index_column = True
                    elif 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' in self.hyperparams['detect_semantic_types'] \
                            and column_name == 'd3mIndex':
                        assert 'https://metadata.datadrivendiscovery.org/types/UniqueKey' not in new_column_semantic_types
                        # It should not be set because there are no columns with this semantic type in whole DataFrame.
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
                        assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
                        new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')
                        if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                            new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
                        has_set_index_column = True

                if 'https://metadata.datadrivendiscovery.org/types/Attribute' in self.hyperparams['detect_semantic_types'] \
                        and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/Attribute' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')

                if 'https://metadata.datadrivendiscovery.org/types/Time' in self.hyperparams['detect_semantic_types'] \
                        and 'http://schema.org/DateTime' in new_column_semantic_types \
                        and 'https://metadata.datadrivendiscovery.org/types/Time' not in new_column_semantic_types:
                    new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Time')

                # Have we added any other semantic type besides unknown type?
                if new_column_semantic_types != normalized_column_semantic_types:
                    if self.hyperparams['remove_unknown_type'] and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in new_column_semantic_types:
                        new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UnknownType')

            new_column_semantic_types_set = set(new_column_semantic_types)
            column_semantic_types_set = set(column_semantic_types)

            fitted_add_semantic_types.append(sorted(new_column_semantic_types_set - column_semantic_types_set))
            fitted_remove_semantic_types.append(sorted(column_semantic_types_set - new_column_semantic_types_set))

        assert len(fitted_add_semantic_types) == len(columns_to_use)
        assert len(fitted_remove_semantic_types) == len(columns_to_use)

        return fitted_add_semantic_types, fitted_remove_semantic_types

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        assert self._add_semantic_types is not None
        assert self._remove_semantic_types is not None

        columns_to_use, output_columns = self._produce_columns(inputs, self._add_semantic_types, self._remove_semantic_types)

        if self.hyperparams['replace_index_columns'] and self.hyperparams['return_result'] == 'append':
            assert len(columns_to_use) == len(output_columns)

            index_columns = inputs.metadata.get_index_columns()

            index_columns_to_use = []
            other_columns_to_use = []
            index_output_columns = []
            other_output_columns = []
            for column_to_use, output_column in zip(columns_to_use, output_columns):
                if column_to_use in index_columns:
                    index_columns_to_use.append(column_to_use)
                    index_output_columns.append(output_column)
                else:
                    other_columns_to_use.append(column_to_use)
                    other_output_columns.append(output_column)

            outputs = base_utils.combine_columns(inputs, index_columns_to_use, index_output_columns, return_result='replace', add_index_columns=self.hyperparams['add_index_columns'])
            outputs = base_utils.combine_columns(outputs, other_columns_to_use, other_output_columns, return_result='append', add_index_columns=self.hyperparams['add_index_columns'])
        else:
            outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query_column(column_index)

        semantic_types = column_metadata.get('semantic_types', [])

        # We detect only on columns which have no semantic types or
        # where it is explicitly set as unknown.
        if not semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in semantic_types:
            return True

        # A special case to handle setting "https://metadata.datadrivendiscovery.org/types/TrueTarget".
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being parsed.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can parsed. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _produce_columns(
        self, inputs: Inputs,
        add_semantic_types: typing.List[typing.List[str]],
        remove_semantic_types: typing.List[typing.List[str]],
    ) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:
        columns_to_use = self._get_columns(inputs.metadata)

        assert len(add_semantic_types) == len(remove_semantic_types), (len(add_semantic_types), len(remove_semantic_types))

        if len(columns_to_use) != len(add_semantic_types):
            raise exceptions.InvalidStateError("Producing on a different number of columns than fitting.")

        output_columns = []

        for column_index, column_add_semantic_types, column_remove_semantic_types in zip(columns_to_use, add_semantic_types, remove_semantic_types):
            output_column = inputs.select_columns([column_index])

            for remove_semantic_type in column_remove_semantic_types:
                output_column.metadata = output_column.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 0), remove_semantic_type)
            for add_semantic_type in column_add_semantic_types:
                output_column.metadata = output_column.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), add_semantic_type)

            output_columns.append(output_column)

        assert len(output_columns) == len(columns_to_use)

        return columns_to_use, output_columns

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

    def _is_unique_key(self, input_column: container.DataFrame) -> bool:
        column_values = input_column.iloc[:, 0]

        # There should be at least one row. This prevents a degenerate case
        # where we would mark a column of no rows as a unique key column.
        # (Otherwise we also get division by zero below.)
        if not len(column_values):
            return False

        # Here we look at every value as-is. Even empty strings and other missing/nan values.
        if any(input_column.duplicated()):
            return False

        return True

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                add_semantic_types=None,
                remove_semantic_types=None,
            )

        return Params(
            add_semantic_types=self._add_semantic_types,
            remove_semantic_types=self._remove_semantic_types,
        )

    def set_params(self, *, params: Params) -> None:
        self._add_semantic_types = params['add_semantic_types']
        self._remove_semantic_types = params['remove_semantic_types']
        self._fitted = all(param is not None for param in params.values())

