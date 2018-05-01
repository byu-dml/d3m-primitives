from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase
from d3m import metadata
from d3m.metadata.hyperparams import Hyperparams
from d3m.metadata.params import Params
from d3m import container
import numpy as np
import pandas


__primitive_version__ = '0.1.0'
__package_version__ = '0.2.0'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(Params):
    column_vals: container.list.List[pandas.core.series.Series]

class RandomSamplingImputer(FeaturizationLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    A primitive which takes raw data and imputes missing values for each column by randomly sampling from the existing values of that column.
    If a column has no existing values (aka a completely empty column), the column is ignored and remains in the dataset unimputed"
    """

    metadata = metadata.base.PrimitiveMetadata({
        'id': 'ebfeb6f0-e366-4082-b1a7-602fd50acc96',
        'version': f'v{__primitive_version__}',
        'name': 'Random Sampling Imputer',
        'source': {
            'name': 'byu-dml',
            'contact': 'https://github.com/byu-dml/d3m-primitives'
        },
        'installation': [
            {
                'type': metadata.base.PrimitiveInstallationType.PIP,
                'package': 'byudml',
                'version': __package_version__
            }
        ],
        'location_uris': [
            'https://github.com/byu-dml/d3m-primitives/blob/master/byu_dml/imputer/random_sampling_imputer.py'
        ],
        'python_path': 'd3m.primitives.byudml.imputer.RandomSamplingImputer',
        'primitive_family': metadata.base.PrimitiveFamily.DATA_PREPROCESSING,
        'algorithm_types': [
            metadata.base.PrimitiveAlgorithmType.IMPUTATION
        ],
        'effects': [
            # not the case if empty columns are just ignored
            metadata.base.PrimitiveEffects.NO_MISSING_VALUES
        ]
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int=0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed = random_seed)
        self._column_vals: container.list.List[container.list.List] = None
        self._random_state = np.random.RandomState(self.random_seed)
        self._training_inputs: Inputs = None
        self._fitted: bool = False

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError('Missing training data.')

        self._column_vals = []
        dataframe = self._training_inputs
        for feature in dataframe:
            dropped_nan_series = dataframe[feature].dropna(axis=0, how='any')
            self._column_vals.append(dropped_nan_series)
        self._fitted = True

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise ValueError("Calling produce before fitting.")
        
        dataframe = inputs
        for i in range(len(dataframe.columns)):
            feature_series = dataframe.iloc[:,i]
            col = feature_series.as_matrix()
            num_nan = np.sum(feature_series.isnull())
            # ignores empty columns
            if len(self._column_vals[i]) > 0:
                col[feature_series.isnull()] = self._random_state.choice(self._column_vals[i], num_nan)
            else:
                self.logger.warning('\nWARNING: %s column is completely empty - no values to impute.  This column contains no information and should be dropped by another primitive.', dataframe.columns[i])
        outputs = dataframe

        outputs.metadata = outputs.metadata.update((), {
            'schema': metadata.base.CONTAINER_SCHEMA_VERSION,
            'structural_type': type(outputs)
        }, source=self)

        return CallResult(outputs)

    def get_params(self) -> Params:
        return Params(column_vals=self._column_vals)

    def set_params(self, *, params: Params) -> None:
        self._column_vals = params['column_vals']




