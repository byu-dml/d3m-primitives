import os
import unittest

from d3m import index, runtime as runtime_module

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer

from tests import utils


DATASETS_DIR = '/datasets/seed_datasets_current'
PIPELINES_DIR = './pipelines'
CLASSIFICATION_PIPELINE_FILENAMES = ['1bee8eae-b571-4f49-90ef-dc3e20f56537.json']
REGRESSION_PIPELINE_FILENAMES = ['44541f7f-27c4-4b4d-ab7e-608b2a2421c6.json']
DATA_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'fixed-split-tabular-split.yml')
SCORING_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'scoring.yml')


class TestMetafeatureExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        index.register_primitive(RandomSamplingImputer.metadata.query()['python_path'], RandomSamplingImputer)

        cls.classification_dataset_util = utils.D3MDatasetUtil(DATASETS_DIR, '185_baseball')
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
