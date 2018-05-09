from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
import os
print('importing common primitives package...')
from d3m.primitives.datasets import DatasetToDataFrame
print('done')

def load_dataset():
	dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', '38_sick_dataset', 'datasetDoc.json'))
	dataset = Dataset.load(f'file://{dataset_doc_path}')
	dataframe = DatasetToDataFrame(hyperparams=None).produce(inputs=dataset).value
	return dataframe