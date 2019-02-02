import typing
import json
import copy
import os
import pandas as pd
from typing import List
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from d3m.container.pandas import DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from metalearn.metafeatures.metafeatures import Metafeatures

__primitive_version__ = "0.4.2"
__package_version__ = "0.6.1"

Inputs = DataFrame
Outputs = DataFrame
class Hyperparams(hyperparams.Hyperparams):

    # This hyperparam takes precedence for determining which metafeatures to compute.
    # If set to ALL, the results will contain all metafeatures computed by the metalearn package (full list can be found in 'metalearn_to_d3m_map.json')
    # If set to CUSTOM, then 'metafeatures_to_compute' hyperparam determines which metafeatures to compute.
    # If set to INEXPENSIVE, the following 7 d3m metafeatures will not be computed:
    #       canonical_correlation
    #       mutual_information_of_numeric_attributes
    #       equivalent_number_of_categorical_attributes
    #       equivalent_number_of_numeric_attributes
    #       categorical_noise_to_signal_ratio
    #       numeric_noise_to_signal_ratio
    #       knn_1_neighbor
    # In reality, a metafeature being expensive/inexpensive depends largley on the nature of the dataset, but these 14 were found to take the longest on a small sample
    metafeature_subset = hyperparams.Enumeration[str](
        values=['INEXPENSIVE', 'CUSTOM', 'ALL'],
        default='INEXPENSIVE',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'],
        description='Subset of available metafeatures the user wishes to compute.  If CUSTOM is chosen, specific list of metafeature names must be provided to \'metafeatures_to_compute\' hyperparam.'
    )

    # This hyperparam is ignored unless 'metafeature_subset' is set to CUSTOM, in which case only the metafeatures in this list are computed
    # Metafeatures must be listed by name as they appear in the 'data_metafeatures' property of the schema at https://gitlab.com/datadrivendiscovery/d3m/blob/devel/d3m/metadata/schemas/v0/definitions.json
    metafeatures_to_compute = hyperparams.Hyperparameter[List[str]](
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/MetafeatureParameter'],
        description='Custom list of specific metafeatures to compute by name.  Only used if \'metafeature_subset\' hyperparam is set to \'CUSTOM\''
    )


class MetafeatureExtractor(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    A primitive which takes a DataFrame and computes metafeatures on the data.  
    Target column is identified by being labled with 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in 'semantic_types' metadata.
    Otherwise primitive assumes there is no target column and only metafeatures that do not invole targets are returned.
    If Dataframe metadata does not include semantic type labels for each column, columns will be classified as CATEGORICAL or NUMERIC according
    to their dtype: int and float are NUMERIC, all others are CATEGORICAL.
    Metafeatures are stored in the metadata object of the DataFrame, and the DataFrame itself is returned unchanged
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_base.PrimitiveMetadata({
        "id": "28d12214-8cb0-4ac0-8946-d31fcbcd4142",
        "version": __primitive_version__,
        "name": "Dataset Metafeature Extraction",
        "source": {
            "name": "byu-dml",
            "contact": "mailto:bjschoenfeld@gmail.com",
            "uris": [
                "https://github.com/byu-dml/d3m-primitives"
            ]
        },
        "installation": [
            {
                "type": metadata_base.PrimitiveInstallationType.PIP,
                "package": "byudml",
                "version": __package_version__
            }
        ],
        'location_uris': [
            'https://github.com/byu-dml/d3m-primitives/blob/master/byu_dml/metafeature_extraction/metafeature_extraction.py'
        ],
        "python_path": "d3m.primitives.metafeature_extraction.metafeature_extractor.BYU",
        "primitive_family": metadata_base.PrimitiveFamily.METAFEATURE_EXTRACTION,
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
            metadata_base.PrimitiveAlgorithmType.CANONICAL_CORRELATION_ANALYSIS,
            metadata_base.PrimitiveAlgorithmType.INFORMATION_ENTROPY,
            metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
            metadata_base.PrimitiveAlgorithmType.SIGNAL_TO_NOISE_RATIO,
            metadata_base.PrimitiveAlgorithmType.STATISTICAL_MOMENT_ANALYSIS
        ],
    })

    _mapping_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metalearn_to_d3m_map.json')

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    # parse metadata to extract column semantic types and identify the target column
    def _get_column_metadata(self, metadata, data):
        column_types = {}
        target_col_name = None
        redacted_target_col_name = None
        for col_pos in range(len(data.columns)):
            column_metadata = metadata.query_column(col_pos)
            semantic_types = column_metadata.get('semantic_types', [])
            column_name = column_metadata.get('name', data.columns[col_pos])
            if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types:
                target_col_name = column_name
            elif 'https://metadata.datadrivendiscovery.org/types/RedactedTarget' in semantic_types:
                redacted_target_col_name = column_name
            if 'http://schema.org/Float' in semantic_types or "http://schema.org/Integer" in semantic_types and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in semantic_types:
                column_types[column_name] = Metafeatures.NUMERIC
                actual_type = str(data[column_name].dtype)
                if 'int' not in actual_type and 'float' not in actual_type:
                    data[column_name] = pd.to_numeric(data[column_name])
            else:
                column_types[column_name] = Metafeatures.CATEGORICAL
        return column_types, target_col_name, redacted_target_col_name

    def _d3m_metafeature_name_to_metalearn_functions(self, d3m_metafeature_name):
        metalearn_functions = []
        mapping = json.load(open(self._mapping_file_path))
        for function_name, properties in mapping.items():
            metafeature_name = properties['data_metafeatures_path'].split('.')[0]
            if metafeature_name == d3m_metafeature_name:
                metalearn_functions.append(function_name)
        return metalearn_functions

    # recursively adds a value to a dictionary given a series of one or more keys
    def _place_value(self, dictionary, path, value):
        if len(path)==0:
            return value
        sub_dict = dictionary.get(path[0], {})
        dictionary[path[0]] = self._place_value(sub_dict, path[1:], value)
        return dictionary

    # parses the mapping file to obtain a list of all the metalearn metafeatures that are classified as inexpensive
    def _get_inexpensive_subset(self):
        inexpensive_subset = []
        mapping = json.load(open(self._mapping_file_path))
        for key, value in mapping.items():
            if value['computation_time']=='inexpensive':
                d3m_metafeature_name = value['data_metafeatures_path'].split('.')[0]
                if d3m_metafeature_name not in inexpensive_subset:
                    inexpensive_subset.append(d3m_metafeature_name)
        return inexpensive_subset

    # returns the user's desired metafeature set according to hyperparam object
    def _get_metafeatures_to_compute(self):
        if self.hyperparams['metafeature_subset']=='CUSTOM':
            return self.hyperparams['metafeatures_to_compute']
        elif self.hyperparams['metafeature_subset']=='INEXPENSIVE':
            return self._get_inexpensive_subset()
        elif self.hyperparams['metafeature_subset']=='ALL':
            return None

    def _get_landmarking_metafeatures(self):
        landmarking_mfs = []
        mapping = json.load(open(self._mapping_file_path))
        for key, value in mapping.items():
            if 'landmarking' in value:
                if value['landmarking']==True:
                    landmarking_mfs.append(key)
        return landmarking_mfs

    # set the 'primitive' and 'random_seed' fields for metafeatures who's results could vary depending on implementation
    def _set_implementation_fields(self, data_metafeatures, data_metafeatures_path):
        landmarking_name = data_metafeatures_path[0]
        if landmarking_name not in data_metafeatures:
            primitive_field_path = [landmarking_name, 'primitive']
            random_seed_field_path = [landmarking_name, 'random_seed']
            primitive_field_val = {'id': self.metadata.query()['id'], 'version': __primitive_version__, 'python_path': self.metadata.query()['python_path'], 'name': self.metadata.query()['name']}
            if 'digest' in self.metadata.query():
                primitive_field_val['digest'] = self.metadata.query()['digest']
            random_seed_field_val = self.random_seed
            data_metafeatures = self._place_value(data_metafeatures, primitive_field_path, primitive_field_val)
            data_metafeatures = self._place_value(data_metafeatures, random_seed_field_path, random_seed_field_val)
        return data_metafeatures

    # populate metadata with metafeatures and return it
    def _populate_metadata(self, metafeatures, metadata, ):
        dataframe_metadata = dict(metadata.query((),))
        data_metafeatures = dataframe_metadata.get('data_metafeatures', {})
        mapping = json.load(open(self._mapping_file_path))
        for column_name in metafeatures.columns:
            if column_name[-4:] != 'Time':
                data_metafeatures_path = mapping[column_name]['data_metafeatures_path'].split(".")
                metafeature_val = metafeatures[column_name][0]
                if pd.notna(metafeature_val) and metafeature_val != Metafeatures.TIMEOUT and metafeature_val != Metafeatures.NO_TARGETS:
                    if column_name in self._get_landmarking_metafeatures():
                        data_metafeatures = self._set_implementation_fields(data_metafeatures, data_metafeatures_path)
                    if mapping[column_name]['required_type']=='integer':
                        metafeature_val = int(metafeature_val)
                    data_metafeatures = self._place_value(data_metafeatures, data_metafeatures_path, metafeature_val)
        dataframe_metadata['data_metafeatures'] = data_metafeatures
        if 'schema' not in dataframe_metadata:
            dataframe_metadata['schema'] = 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json'
        if 'structural_type' not in dataframe_metadata:
            dataframe_metadata['structural_type'] = DataFrame
        metadata = metadata.update((), dataframe_metadata)
        return metadata

    # given a d3m Dataframe, return it with the computed metafeatures (specified by the hyperparam) added to it's metadata
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not isinstance(inputs, DataFrame):
            raise ValueError("inputs must be an instance of 'd3m.container.pandas.DataFrame'")
        metadata = self._produce(inputs.metadata, copy.copy(inputs), timeout)

        inputs.metadata = metadata.set_for_value(inputs)

        return CallResult(inputs)


    def _produce(self, metadata, data, timeout):
        # get semantic types of columns and target column name
        column_types, target_col_name, redacted_target_col_name = self._get_column_metadata(metadata, data)
        if len(column_types) != len(data.columns):
            column_types = None

        # separate features from targets, and drop d3mIndex column if present
        if target_col_name is not None:
            target_series = data[target_col_name]
            data.drop(target_col_name, axis=1, inplace=True)
        elif redacted_target_col_name is not None:
            target_series = None
            data.drop(redacted_target_col_name, axis=1, inplace=True)
            if column_types is not None:
                del column_types[redacted_target_col_name]
        else:
            target_series = None
            self.logger.warning(f'\nWARNING: Found no column labled with \'https://metadata.datadrivendiscovery.org/types/TrueTarget\' or \'https://metadata.datadrivendiscovery.org/types/RedactedTarget\'in metadata.  Program will continue with the assumption that there is no target column, and only store requested metafeatures that do not involve targets\n')
        if 'd3mIndex' in data.columns:
            data.drop('d3mIndex', axis=1, inplace=True)
            if column_types is not None:
                del column_types['d3mIndex']

        # translate d3m metafeature names to metalearn names
        d3m_metafeatures_to_compute = self._get_metafeatures_to_compute()
        if d3m_metafeatures_to_compute is not None:
            metalearn_metafeatures_to_compute = []
            for mf in d3m_metafeatures_to_compute:
                metalearn_functions = self._d3m_metafeature_name_to_metalearn_functions(mf)
                metalearn_metafeatures_to_compute.extend(metalearn_functions)
        else:
            metalearn_metafeatures_to_compute = None

        # compute metafeatures and return in metadata
        metafeatures = Metafeatures().compute(data, target_series, column_types=column_types, metafeature_ids=metalearn_metafeatures_to_compute, seed=self.random_seed, timeout=timeout)
        metafeature_df = pd.DataFrame.from_dict([{mf: metafeatures[mf][Metafeatures.VALUE_KEY] for mf in metafeatures}])
        metadata = self._populate_metadata(metafeature_df, metadata)
        return metadata