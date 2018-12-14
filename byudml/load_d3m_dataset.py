from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
import os
from d3m.primitives.datasets import DatasetToDataFrame
from d3m.primitives.data import ColumnParser

def load_dataset():
	dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', '38_sick_dataset', 'datasetDoc.json'))
	dataset = Dataset.load(f'file://{dataset_doc_path}')
	hyperparams_class = DatasetToDataFrame.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
	dataframe = DatasetToDataFrame(hyperparams=hyperparams_class.defaults()).produce(inputs=dataset).value
	hyperparams_class = ColumnParser.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
	dataframe = ColumnParser(hyperparams=hyperparams_class.defaults()).produce(inputs=dataframe).value
	return dataframe