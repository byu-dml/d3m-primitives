from setuptools import setup, find_packages

__version__ = '0.5.4'

setup(
	name="byudml",
	packages = find_packages(),
	version = __version__,
	description = "A collection of DARPA D3M primitives developed by BYU",
	author = "Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen",
	url = "https://github.com/byu-dml/d3m-primitives",
	include_package_data=True,
	keywords = ["metalearning", "metafeature", "machine learning", "metalearn", "d3m_primitive"],
	install_requires = [
		"metalearn==0.4.7",
		"numpy<=1.14.3",
		"pandas<=0.22.0"
	],
	entry_points = {
    	'd3m.primitives': [
    		'byudml.imputer = workaround_ignore_error',
    		'byudml.metafeature_extraction = workaround_ignore_error',
        	'byudml.imputer.RandomSamplingImputer = byudml.imputer.random_sampling_imputer:RandomSamplingImputer',
        	'byudml.metafeature_extraction.MetafeatureExtractor = byudml.metafeature_extraction.metafeature_extraction:MetafeatureExtractor'
    	]
	}
)