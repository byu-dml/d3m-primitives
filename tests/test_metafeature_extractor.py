import os
import unittest

from d3m import index, runtime as runtime_module

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor

from tests import utils


DATASETS_DIR = '/datasets/seed_datasets_current'
PIPELINES_BASE_DIR = './pipelines'
PIPELINES_DIR = os.path.join(PIPELINES_BASE_DIR, 'metafeature_extractor')
CLASSIFICATION_PIPELINE_FILENAMES = [
    'b32b9af1-34b4-437b-ad83-650f7df10acb.json'
]
REGRESSION_PIPELINE_FILENAMES = [
    '3013ad40-7c51-4991-b0fb-dbec65607979.json'
]
DATA_PIPELINE_PATH = os.path.join(PIPELINES_BASE_DIR, 'fixed-split-tabular-split.yml')
SCORING_PIPELINE_PATH = os.path.join(PIPELINES_BASE_DIR, 'scoring.yml')


class TestMetafeatureExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        index.register_primitive(RandomSamplingImputer.metadata.query()['python_path'], RandomSamplingImputer)
        index.register_primitive(MetafeatureExtractor.metadata.query()['python_path'], MetafeatureExtractor)

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
