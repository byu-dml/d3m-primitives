from setuptools import setup, find_packages
from byu_dml.imputer import __version__

setup(
	name="byudml",
	packages = find_packages(),
	version = __version__,
	description = "A collection of DARPA D3M primitives developed by BYU",
	author = "Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen",
	url = "https://github.com/byu-dml/d3m-primitives",
	keywords = ["d3m_primitive"],
	install_requires = [
		"numpy",
		"pandas"
	],
	entry_points = {
    	'd3m.primitives': [
        	'byudml.imputer.RandomSamplingImputer = byu_dml.imputer:RandomSamplingImputer'
    	],
	}
)