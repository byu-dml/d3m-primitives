from setuptools import setup, find_packages

from byudml import __version__
from byudml import __imputer_path__
from byudml import __metafeature_path__


setup(
    name='byudml',
    packages = find_packages(include=['byudml', 'byudml.*']),
    version = __version__,
    description = 'A collection of DARPA D3M primitives developed by BYU',
    author = 'Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen',
    url = 'https://github.com/byu-dml/d3m-primitives',
    include_package_data=True,
    keywords = ['metalearning', 'metafeature', 'machine learning', 'metalearn', 'd3m_primitive'],
    install_requires = [
        'd3m',
        'metalearn==0.5.4',
        'numpy<=1.17.3',
        'pandas<=0.25.2'
    ],
    entry_points = {
        'd3m.primitives': [
            '{} = byudml.imputer.random_sampling_imputer:RandomSamplingImputer'.format(".".join(__imputer_path__.split(".")[2:])),
            '{} = byudml.metafeature_extraction.metafeature_extraction:MetafeatureExtractor'.format(".".join(__metafeature_path__.split(".")[2:]))
        ]
    }
)
