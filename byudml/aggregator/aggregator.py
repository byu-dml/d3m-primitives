import typing

import pandas as pd
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

from byudml import __version__ as __package_version__
from byudml import __aggregator_path__, __aggregator_version__


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    output_semantic_types = hyperparams.Hyperparameter[typing.List[str]](
        default=['https://metadata.datadrivendiscovery.org/types/Attribute'],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls which semantic types will be assigned to the resulting mean of the dataframe.",
    )
    use_mode_column_name = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=(
            "Whether to use the most common column name from the inputs as the column "
            "name of the output. Useful when this primitive is used for ensembling, so "
            "the name of the ensembled output column can be the target name. If False, "
            "the name of the aggregator hyperparam will be used instead (e.g. 'mean', 'mode')."
        )
    )
    aggregator = hyperparams.Enumeration(
        values=['mean', 'mode'],
        default='mode',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=(
            "Which row-wise aggregation to take of the input dataframe. "
            "`mean` will only consider numeric values. `mode` will consider "
            "all datatypes. If modes are taken across both numeric and non-numeric "
            "columns, note that the numeric structural types of the numeric modes "
            "will be lost, since the resulting mode column will hold both numeric and "
            "non-numeric (e.g. strings) values."
        )
    )


class Aggregator(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    """
    Takes a row-wise aggregation of the input data. Can be useful for ensembling
    (e.g. taking the mode for ensembling classifiers or the mean for ensembling
    regressors).
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': '28030b64-f666-4175-af5a-0dd2b218343d',
        'version': __aggregator_version__,
        'name': "Mean",
        'keywords': [
            'mean', 'statistic', 'ensembling', 'aggregate', 'aggregator',
            'aggregation', 'ensemble', 'mode', 'average', 'vote', 'voting'
        ],
        'source': {
            'name': 'byu-dml',
            'contact': 'mailto:bjschoenfeld@gmail.com',
            'uris': [
                'https://github.com/byu-dml/d3m-primitives'
            ]
        },
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'byudml',
                'version': __package_version__
            }
        ],
        'location_uris': [
            'https://github.com/byu-dml/d3m-primitives/blob/master/byu_dml/aggregator/aggregator.py'
        ],
        'python_path': __aggregator_path__,
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.AGGREGATE_FUNCTION,
            metadata_base.PrimitiveAlgorithmType.ENSEMBLE_LEARNING
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION
    })

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        aggregator = self.hyperparams['aggregator']

        if self.hyperparams['use_mode_column_name']:
            col_names = list(inputs.columns)
            most_common_col_name = max(col_names, key=col_names.count)
            output_name = most_common_col_name
        else:
            output_name = aggregator

        if aggregator == 'mean':
            series_output = inputs.mean(axis=1, numeric_only=True)
        elif aggregator == 'mode':
            # Pandas outputs more than one column when a row
            # has more than one mode so we just use the first
            # column and get the first mode for all.
            series_output = inputs.mode(axis=1).iloc[:,0]
        
        df_output = pd.DataFrame({output_name: series_output})
        outputs = container.DataFrame(df_output, generate_metadata=True)

        for semantic_type in self.hyperparams['output_semantic_types']:
            outputs.metadata = outputs.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), semantic_type)
        
        return base.CallResult(outputs)
