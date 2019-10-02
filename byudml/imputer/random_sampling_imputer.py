import numpy as np

from d3m import container
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase

from byudml import __version__ as __package_version__
from byudml import __imputer_path__, __imputer_version__


Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    drop_cols_all_unknown_vals = hyperparams.Enumeration[bool](
        values=[True, False],
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],

    )


class Params(params.Params):

    known_values: container.list.List


class RandomSamplingImputer(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):

    """
    This imputes missing values in a DataFrame by sampling known values from
    each column independently. If the training data has no known values in a
    particular column, no values are imputed. The default behavior is to
    remove columns with no known values. These columns can be retained by
    setting the drop_cols_all_unknown_vals hyperparameter to True.
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
            'https://github.com/byu-dml/d3m-primitives/blob/master/byu_dml/imputer/random_sampling_imputer.py'
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

        # operate on columns by index, not name
        self._known_values = []
        for col_name in self._training_inputs.columns:
            self._known_values.append(self._training_inputs[col_name].dropna(axis=0, how='any'))

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise ValueError('Calling produce before fitting.')

        if inputs.shape[1] != len(self._known_values):
            raise ValueError(
                'The number of input columns does not match the training data: {} != {}'.format(
                    inputs.shape[1], len(self._known_values)
                )
            )

        drop_cols_all_unknown_vals: bool = self.hyperparams['drop_cols_all_unknown_vals']
        if drop_cols_all_unknown_vals:
            columns_to_remove: list = []  # Columns that have no known values in them need to be dropped

        for i, col_name in enumerate(inputs):
            # ignores empty columns
            if len(self._known_values[i]) > 0:
                inputs_isnull = inputs[col_name].isnull()
                n_missing = sum(inputs_isnull)
                if n_missing > 0:
                    inputs[col_name][inputs_isnull] = self._random_state.choice(
                        self._known_values[i], n_missing, replace=True
                    )
                    # TODO: update column metadata?
            else:
                if drop_cols_all_unknown_vals:
                    # Remove the column that has no values
                    columns_to_remove.append(i)

        if drop_cols_all_unknown_vals:
            inputs: Inputs = inputs.remove_columns(columns_to_remove)

        # TODO: update global metadata if any values were imputed?
        # inputs.metadata = inputs.metadata.update((), {
        #     'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        #     'structural_type': type(outputs)
        # })

        return CallResult(inputs)

    def get_params(self) -> Params:
        return Params(known_values=self._known_values)

    def set_params(self, *, params: Params) -> None:
        self._known_values = params['known_values']
