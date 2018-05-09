import typing
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from d3m.container.pandas import DataFrame
from d3m import metadata

from metalearn.metafeatures.metafeatures import Metafeatures

import pandas as pd

__primitive_version__ = "0.3.0"
__package_version__ = "0.4.0"

Inputs = DataFrame
Outputs = DataFrame
class Hyperparams(metadata.hyperparams.Hyperparams):
    # could be used to determine which metafeatures to compute
    pass

class MetafeatureExtractor(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    A primitive which takes a DataFrame with one target column and computes metafeatures on the data set.
    DataFrame metadata object must have 'https://metadata.datadrivendiscovery.org/types/Target' in 'semantic_types' metadata for the target column,
    or it will not be identified and an error will be raised.
    Metafeatures are returned as a single-row DataFrame, with a column for each metafeature
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata.base.PrimitiveMetadata({
        "id": "28d12214-8cb0-4ac0-8946-d31fcbcd4142",
        "version": f"v{__primitive_version__}",
        "name": "Dataset Metafeature Extraction",
        "source": {
            "name": "byu-dml",
            "contact": "https://github.com/byu-dml/d3m-primitives"
        },
        "installation": [
            {
                "type": metadata.base.PrimitiveInstallationType.PIP,
                "package": "byudml",
                "version": __package_version__
            }
        ],
        'location_uris': [
            'https://github.com/byu-dml/d3m-primitives/blob/master/byu_dml/metafeature_extraction/metafeature_extraction.py'
        ],
        "python_path": "d3m.primitives.byudml.metafeature_extraction.MetafeatureExtractor",
        "primitive_family": metadata.base.PrimitiveFamily.METAFEATURE_EXTRACTION,
        "algorithm_types": [
            metadata.base.PrimitiveAlgorithmType.DATA_PROFILING,
            metadata.base.PrimitiveAlgorithmType.CANONICAL_CORRELATION_ANALYSIS,
            metadata.base.PrimitiveAlgorithmType.INFORMATION_ENTROPY,
            metadata.base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
            metadata.base.PrimitiveAlgorithmType.SIGNAL_TO_NOISE_RATIO,
            metadata.base.PrimitiveAlgorithmType.STATISTICAL_MOMENT_ANALYSIS
        ],
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not isinstance(inputs, DataFrame):
            raise ValueError("inputs must be an instance of 'd3m.container.pandas.DataFrame'")

        # parse dataframe metadata to extract column dtypes and identify the target column
        column_types = {}
        target_col_name = None
        for col_pos in inputs.metadata.get_elements((metadata.base.ALL_ELEMENTS,)):
            column_metadata = inputs.metadata.query((metadata.base.ALL_ELEMENTS, col_pos))
            semantic_types = column_metadata.get('semantic_types', [])
            column_name = column_metadata.get('name', [])
            if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                target_col_name = column_name
            if 'http://schema.org/Float' in semantic_types or "http://schema.org/Integer" in semantic_types and 'https://metadata.datadrivendiscovery.org/types/CategoricalData' not in semantic_types:
                column_types[column_name] = Metafeatures.NUMERIC
            else:
                column_types[column_name] = Metafeatures.CATEGORICAL

        # convert each semantically numeric column to actual numeric dtype
        for col_name, dtype in column_types.items():
            if dtype == Metafeatures.NUMERIC:
                inputs[col_name] = pd.to_numeric(inputs[col_name])

        # separate features from targets, and drop d3mIndex column if present
        if target_col_name==None:
            raise ValueError('inputs metadata must contain \'https://metadata.datadrivendiscovery.org/types/Target\' in \'semantic_types\' of target column.')
        target_series = inputs[target_col_name]
        if 'd3mIndex' in inputs.columns:
            dataframe = inputs.drop(['d3mIndex', target_col_name], axis=1)
            del column_types['d3mIndex']
        else:
            dataframe = inputs.drop(target_col_name, axis=1)

        # compute metafeatures
        metafeatures = Metafeatures().compute(dataframe, target_series, column_types, seed=self.random_seed, timeout=timeout)
        metafeatures_df = DataFrame(metafeatures)

        return CallResult(metafeatures_df)