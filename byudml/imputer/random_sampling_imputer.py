import numpy as np
import typing

from d3m import container, exceptions as d3m_exceptions
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from byudml import __imputer_path__, __imputer_version__
from byudml import __version__ as __package_version__


Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    drop_missing_values = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Determines whether to drop columns containing missing values.'
    )
    how = hyperparams.Enumeration[str](
        values=['all', 'any'],
        default='all',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Determines how to drop missing values. If "all", drops columns where all values are missing. If "any", drops columns where any values are missing (note no imputation is performed).'
    )


class Params(params.Params):
    known_values: typing.Sequence[typing.Any]
    drop_cols: typing.Sequence[bool]
    drop_col_indices: typing.Sequence[int]


class RandomSamplingImputer(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    This imputes missing values in a DataFrame by sampling known values from each column independently. If the training
    data has no known values in a particular column, no values are imputed. Alternatively, columns with missing values
    can be dropped. By default columns of all missing values are dropped.
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': 'ebfeb6f0-e366-4082-b1a7-602fd50acc96',
        'version': __imputer_version__,
        'name': 'Random Sampling Imputer',
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
            'https://github.com/byu-dml/d3m-primitives/blob/master/byudml/imputer/random_sampling_imputer.py'
        ],
        'python_path': __imputer_path__,
        'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.IMPUTATION
        ],
        'effects': [
            # not the case if empty columns are just ignored
            metadata_base.PrimitiveEffect.NO_MISSING_VALUES
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed = random_seed)
        self._random_state = np.random.RandomState(self.random_seed)

        self._fitted: bool = False
        self._training_inputs: Inputs = None
        self._known_values = None
        self._drop_cols = None
        self._drop_col_indices = None

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._fitted = False
        self._training_inputs = inputs
        self._known_values = []
        self._drop_cols = []
        self._drop_col_indices = []

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise d3m_exceptions.MissingValueError('set_training_data must be called before fit')

        # operate on columns by index, not name
        for i, (col_name, col) in enumerate(self._training_inputs.iteritems()):
            drop_col = False
            if self.hyperparams['drop_missing_values']:
                if self.hyperparams['how'] == 'all' and col.isnull().all():
                    drop_col = True
                elif self.hyperparams['how'] == 'any' and col.isnull().any():
                    drop_col = True
            self._drop_cols.append(drop_col)
            if drop_col:
                self._drop_col_indices.append(i)

            col_known_values = None
            if not drop_col:
                col_known_values = col.dropna(axis=0, how='any').tolist()
            self._known_values.append(col_known_values)

        self._fitted = True
        self._training_inputs = None  # free memory

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise d3m_exceptions.PrimitiveNotFittedError('fit must be called before produce')

        if inputs.shape[1] != len(self._known_values):
            raise d3m_exceptions.DimensionalityMismatchError(
                'The number of input columns does not match the training data: {} != {}'.format(
                    inputs.shape[1], len(self._known_values)
                )
            )

        outputs = inputs.copy()
        for i, (col_name, col) in enumerate(inputs.iteritems()):
            if self._drop_cols[i]:
                assert self._known_values[i] is None
            else:
                indices_of_missing_values = col.isnull()
                n_missing = indices_of_missing_values.sum()
                n_known = len(self._known_values[i])
                if n_missing > 0 and n_known > 0:  # k_known == 0 implies drop_missing_values == False
                    outputs.loc[indices_of_missing_values, col_name] = self._random_state.choice(
                        self._known_values[i], n_missing, replace=True
                    )
                    # TODO: update column metadata?

        outputs = outputs.remove_columns(self._drop_col_indices)

        # TODO: update global metadata if any values were imputed?

        return CallResult(outputs)

    def get_params(self) -> Params:
        if not self._fitted:
            raise d3m_exceptions.PrimitiveNotFittedError('fit must be called before get_params')
        return Params(known_values=self._known_values, drop_cols=self._drop_cols, drop_col_indices=self._drop_col_indices)

    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._training_inputs = None
        self._known_values = params['known_values']
        self._drop_cols = params['drop_cols']
        self._drop_col_indices = params['drop_col_indices']
