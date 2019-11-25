import os
import unittest

import d3m
from d3m import index, container
from d3m.metadata import base as metadata_base
import numpy as np
import pandas as pd

from byudml.imputer.random_sampling_imputer import Hyperparams, RandomSamplingImputer

from tests import utils


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


class RandomSamplingImputerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        index.register_primitive(RandomSamplingImputer.metadata.query()['python_path'], RandomSamplingImputer)

        cls.classification_dataset_util = utils.D3MDatasetUtil(DATASETS_DIR, '38_sick')
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

    @staticmethod
    def _get_test_data_frame() -> container.pandas.DataFrame:
        return container.pandas.DataFrame(
            {
                'all_known': [1,2,3,4],
                'some_missing': [6,None,10,12],
                'all_missing': [None, None, None, None],
            },
            generate_metadata=True
        )

    @staticmethod
    def _get_imputer(drop_missing_values, how):
        hyperparams = Hyperparams({'drop_missing_values': drop_missing_values, 'how': how})
        return RandomSamplingImputer(hyperparams=hyperparams)

    @staticmethod
    def _get_metadata_cols(metadata) -> list:
        return [metadata.query_column_field(i, 'name') for i in metadata.get_elements([metadata_base.ALL_ELEMENTS])]

    def _test_output_column_names(self, outputs, expected_output_col_names):
        output_col_names = list(outputs.columns)
        self.assertEqual(output_col_names, expected_output_col_names)
        output_metadata_col_names = self._get_metadata_cols(outputs.metadata)
        self.assertEqual(output_metadata_col_names, expected_output_col_names)

    def test_no_drop(self):
        data = self._get_test_data_frame()
        for how in ['all', 'any']:
            imputer = self._get_imputer(False, how)

            imputer.set_training_data(inputs=data)
            imputer.fit()
            result = imputer.produce(inputs=data)

            expected_output_col_names = ['all_known', 'some_missing', 'all_missing']
            self._test_output_column_names(result.value, expected_output_col_names)

            for col_name in data:
                self.assertTrue(result.value[col_name].isin(data[col_name]).all())

    def test_drop_all(self):
        data = self._get_test_data_frame()
        imputer = self._get_imputer(True, 'all')

        imputer.set_training_data(inputs=data)
        imputer.fit()
        result = imputer.produce(inputs=data)

        expected_output_col_names = ['all_known', 'some_missing']
        self._test_output_column_names(result.value, expected_output_col_names)

        self.assertFalse(result.value.isnull().any().any())
        self.assertTrue(result.value['some_missing'].isin(data['some_missing']).all())

    def test_drop_any(self):
        data = self._get_test_data_frame()
        imputer = self._get_imputer(True, 'any')

        imputer.set_training_data(inputs=data)
        imputer.fit()
        result = imputer.produce(inputs=data)

        expected_output_col_names = ['all_known']
        self._test_output_column_names(result.value, expected_output_col_names)

        self.assertFalse(result.value.isnull().any().any())

    def test_params(self):
        data = self._get_test_data_frame()
        imputer = self._get_imputer(True, 'all')

        # test get_params error
        with self.assertRaises(d3m.exceptions.PrimitiveNotFittedError) as cm:
            imputer.get_params()

        imputer.set_training_data(inputs=data)
        imputer.fit()

        # test get_ and set_params without error
        params = imputer.get_params()
        imputer.set_params(params=params)

        # test set_params with new imputer instance without error
        imputer = self._get_imputer(True, 'all')
        imputer.set_params(params=params)
        result = imputer.produce(inputs=data)

        # test imputed data after set_params
        expected_output_col_names = ['all_known', 'some_missing']
        self._test_output_column_names(result.value, expected_output_col_names)

        self.assertFalse(result.value.isnull().any().any())
        self.assertTrue(result.value['some_missing'].isin(data['some_missing']).all())
