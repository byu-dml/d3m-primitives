from setuptools import setup, find_packages

setup(
	name="byudml",
	packages = find_packages(),
	version = "0.4.0",
	description = "A collection of DARPA D3M primitives developed by BYU",
	author = "Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen",
	url = "https://github.com/byu-dml/d3m-primitives",
	keywords = ["metalearning", "metafeature", "machine learning", "metalearn", "d3m_primitive"],
	install_requires = [
		"metalearn==0.4.4",
		"numpy",
		"pandas"
	],
	entry_points = {
    	'd3m.primitives': [
        	'byudml.imputer.RandomSamplingImputer = byudml.imputer:RandomSamplingImputer',
        	'byudml.metafeature_extraction.MetafeatureExtractor = byudml.metafeature_extraction:MetafeatureExtractor'
    	]
	}
)