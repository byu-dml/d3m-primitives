import os
import unittest

from d3m import index, runtime as runtime_module

from byudml.imputer.random_sampling_imputer import RandomSamplingImputer
from byudml.metafeature_extraction.metafeature_extraction import MetafeatureExtractor

from tests import utils


DATASETS_DIR = '/datasets/seed_datasets_current'
PIPELINES_DIR = './pipelines'
PIPELINE_FILENAMES = ['1bee8eae-b571-4f49-90ef-dc3e20f56537.json']
DATA_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'fixed-split-tabular-split.yml')
SCORING_PIPELINE_PATH = os.path.join(PIPELINES_DIR, 'scoring.yml')


class TestMetafeatureExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        index.register_primitive(RandomSamplingImputer.metadata.query()['python_path'], RandomSamplingImputer)
        index.register_primitive(MetafeatureExtractor.metadata.query()['python_path'], MetafeatureExtractor)

        cls.sick_dataset_util = utils.D3MDatasetUtil(DATASETS_DIR, '185_baseball')
        cls.pipeline_paths = []
        for filename in PIPELINE_FILENAMES:
            cls.pipeline_paths.append(os.path.join(PIPELINES_DIR, filename))

    def test_evaluate_impute_pipeline(self):
        for pipeline_path in self.pipeline_paths:
            utils.evaluate_pipeline(
                pipeline_path, DATA_PIPELINE_PATH,
                self.sick_dataset_util.data_splits_path, SCORING_PIPELINE_PATH,
                self.sick_dataset_util.dataset_doc_path,
                self.sick_dataset_util.problem_path,
                'pipeline_run.yml'
            )
