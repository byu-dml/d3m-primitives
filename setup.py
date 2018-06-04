from setuptools import setup, find_packages

setup(
	name="byudml",
	packages = find_packages(),
	version = "0.5.3",
	description = "A collection of DARPA D3M primitives developed by BYU",
	author = "Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen",
	url = "https://github.com/byu-dml/d3m-primitives",
	include_package_data=True,
	keywords = ["metalearning", "metafeature", "machine learning", "metalearn", "d3m_primitive"],
	install_requires = [
		"metalearn==0.4.6",
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