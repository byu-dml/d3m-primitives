import os
import unittest

from d3m import index, runtime as runtime_module

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor

from tests import utils


DATASETS_DIR = '/datasets/seed_datasets_current'
PIPELINES_DIR = './pipelines'
CLASSIFICATION_PIPELINE_FILENAMES = ['ee7b1517-3547-4689-9e2c-d2f4e3fa5064.json']
REGRESSION_PIPELINE_FILENAMES = ['63347dd2-0565-4ded-a26d-72b1622ddc4c.json']
DATA_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'fixed-split-tabular-split.yml')
SCORING_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'scoring.yml')


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
