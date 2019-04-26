from setuptools import setup, find_packages

from byudml import __version__


setup(
    name='byudml',
    packages = find_packages(),
    version = __version__,
    description = 'A collection of DARPA D3M primitives developed by BYU',
    author = 'Roland Laboulaye, Brandom Schoenfeld, Jarom Christensen',
    url = 'https://github.com/byu-dml/d3m-primitives',
    include_package_data=True,
    keywords = ['metalearning', 'metafeature', 'machine learning', 'metalearn', 'd3m_primitive'],
    install_requires = [
        'd3m',
        'metalearn==0.5.3',
        'numpy<=1.15.4',
        'pandas<=0.23.4'
    ],
    entry_points = {
        'd3m.primitives': [
            'data_preprocessing.random_sampling_imputer.BYU = byudml.imputer.random_sampling_imputer:RandomSamplingImputer',
            'metafeature_extraction.meta_feature_extractor.BYU = byudml.metafeature_extraction.metafeature_extraction:MetafeatureExtractor'
        ]
    }
)
