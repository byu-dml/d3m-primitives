import os
import pandas as pd
import unittest

from d3m import index
from d3m.container.pandas import DataFrame
from d3m.metadata.base import DataMetadata, ALL_ELEMENTS

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.imputer.random_sampling_imputer import Hyperparams
from byudml import strings

from tests import utils
from tests import test_strings


DATASETS_DIR = '/datasets/seed_datasets_current'
PIPELINES_BASE_DIR = 'submission/pipelines'
PIPELINES_DIR = os.path.join(PIPELINES_BASE_DIR, 'random_sampling_imputer')
CLASSIFICATION_PIPELINE_FILENAMES = [
    'f4fe3fcc-45fe-4c85-8845-549e2f466f21.json',
]
REGRESSION_PIPELINE_FILENAMES = [
    '74f5ccb1-053a-46cf-ad7f-005f67a15652.json',
]
DATA_PIPELINE_PATH = os.path.join(PIPELINES_BASE_DIR, 'fixed-split-tabular-split.yml')
SCORING_PIPELINE_PATH = os.path.join(PIPELINES_BASE_DIR, 'scoring.yml')


class TestRandomSamplingImputer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        index.register_primitive(RandomSamplingImputer.metadata.query()['python_path'], RandomSamplingImputer)

        cls.classification_dataset_util = utils.D3MDatasetUtil(DATASETS_DIR, test_strings.CLASSIFICATION_DATASET_NAME)
        cls.classification_pipeline_paths = []
        for filename in CLASSIFICATION_PIPELINE_FILENAMES:
            cls.classification_pipeline_paths.append(os.path.join(PIPELINES_DIR, filename))

        cls.regression_dataset_util = utils.D3MDatasetUtil(DATASETS_DIR, '196_autoMpg')
        cls.regression_pipeline_paths = []
        for filename in REGRESSION_PIPELINE_FILENAMES:
            cls.regression_pipeline_paths.append(os.path.join(PIPELINES_DIR, filename))

    def test_classification(self):
        for pipeline_path in self.classification_pipeline_paths:
            utils.evaluate_pipeline(
                pipeline_path, DATA_PIPELINE_PATH,
                self.classification_dataset_util.data_splits_path, SCORING_PIPELINE_PATH,
                self.classification_dataset_util.dataset_doc_path,
                self.classification_dataset_util.problem_path
            )

    def test_regression(self):
        for pipeline_path in self.regression_pipeline_paths:
            utils.evaluate_pipeline(
                pipeline_path, DATA_PIPELINE_PATH,
                self.regression_dataset_util.data_splits_path, SCORING_PIPELINE_PATH,
                self.regression_dataset_util.dataset_doc_path,
                self.regression_dataset_util.problem_path
            )

    def test_column_with_all_unknown_values(self):
        """
        Tests the Random Sampling Imputer's ability to handle columns that have all unknown values
        :return: None
        """

        data_frame: DataFrame = self._get_test_data_frame()
        original_cols: list = list(data_frame)
        leftover_cols: list = list(data_frame)
        leftover_cols.remove(test_strings.ALL_UNKNOWN_KEY)

        # Test retaining a column with all unknown values
        output_data_frame: DataFrame = self._get_imputer_output(data_frame, drop_cols_all_unknown_vals=False)
        output_cols: list = list(output_data_frame)
        self.assertEqual(output_cols, original_cols)
        self.assertNotEqual(output_cols, leftover_cols)
        metadata_cols: list = self._get_metadata_cols(output_data_frame.metadata)
        self.assertEqual(metadata_cols, output_cols)

        # Test dropping a column with all unknown values
        data_frame: DataFrame = self._get_test_data_frame()
        output_data_frame: DataFrame = self._get_imputer_output(data_frame, drop_cols_all_unknown_vals=True)
        output_cols: list = list(output_data_frame)
        self.assertEqual(output_cols, leftover_cols)
        self.assertNotEqual(output_cols, original_cols)
        self.assertTrue(len(output_cols) < len(original_cols))
        metadata_cols: list = self._get_metadata_cols(output_data_frame.metadata)
        self.assertEqual(metadata_cols, output_cols)

    def test_imputation_works(self):
        """
        Tests to make sure missing values are imputed in columns with some unknown values.
        """
        data_frame: DataFrame = self._get_test_data_frame()

        output_df_no_drop: DataFrame = self._get_imputer_output(data_frame, drop_cols_all_unknown_vals=False)
        self.assertFalse(output_df_no_drop[test_strings.ALL_KNOWN_KEY].isnull().values.any())
        self.assertFalse(output_df_no_drop[test_strings.SOME_KNOWN_KEY].isnull().values.any())
        self.assertTrue(output_df_no_drop[test_strings.ALL_UNKNOWN_KEY].isnull().values.any())

        output_df_with_drop: DataFrame = self._get_imputer_output(data_frame, drop_cols_all_unknown_vals=True)
        self.assertFalse(output_df_no_drop[test_strings.ALL_KNOWN_KEY].isnull().values.any())
        self.assertFalse(output_df_with_drop[test_strings.SOME_KNOWN_KEY].isnull().values.any())

    def _get_imputer_output(self, data_frame: DataFrame, drop_cols_all_unknown_vals: bool) -> DataFrame:
        imputer: RandomSamplingImputer = self._get_imputer(data_frame, drop_cols_all_unknown_vals)
        imputer.fit()
        output_data_frame: DataFrame = imputer.produce(inputs=data_frame).value
        return output_data_frame

    @staticmethod
    def _get_test_data_frame() -> DataFrame:
        nan = float('nan')
        data_frame_dict: dict = {
            test_strings.ALL_KNOWN_KEY: [1, 2, 3, 3],
            test_strings.SOME_KNOWN_KEY: [1, nan, 1, 2],
            test_strings.ALL_UNKNOWN_KEY: [nan, nan, nan, nan]
        }
        data_frame: pd.DataFrame = pd.DataFrame(data_frame_dict)
        data_frame: DataFrame = DataFrame(data_frame, generate_metadata=True)
        return data_frame

    @staticmethod
    def _get_imputer_hyperparams(drop_cols_all_unknown_vals: bool) -> Hyperparams:
        hyperparams: dict = {
            strings.DROP_COLS_ALL_UNKNOWN_VALS_NAME: drop_cols_all_unknown_vals
        }
        hyperparams: Hyperparams = Hyperparams(hyperparams)
        return hyperparams

    def _get_imputer(self, data_frame: DataFrame, drop_cols_all_unknown_vals: bool) -> RandomSamplingImputer:
        hyperparams: Hyperparams = self._get_imputer_hyperparams(drop_cols_all_unknown_vals)
        imputer: RandomSamplingImputer = RandomSamplingImputer(hyperparams=hyperparams)
        imputer.set_training_data(inputs=data_frame)
        return imputer

    @staticmethod
    def _get_metadata_cols(metadata: DataMetadata) -> list:
        col_names: list = []
        for i in metadata.get_elements((list(()) + [ALL_ELEMENTS])):
            col_name: str = metadata.query_column_field(i, test_strings.COL_NAME_KEY)
            col_names.append(col_name)
        return col_names
