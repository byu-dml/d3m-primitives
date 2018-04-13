import typing
import d3m_metadata
from primitive_interfaces.base import CallResult
from primitive_interfaces.featurization import FeaturizationTransformerPrimitiveBase
from metalearn.metafeatures.simple_metafeatures import SimpleMetafeatures
from metalearn.metafeatures.statistical_metafeatures import StatisticalMetafeatures
from metalearn.metafeatures.information_theoretic_metafeatures import InformationTheoreticMetafeatures
from metalearn.metafeatures.landmarking_metafeatures import LandmarkingMetafeatures

import pandas as pd

__version__ = "0.2.0"

Inputs = d3m_metadata.container.pandas.DataFrame
Outputs = d3m_metadata.container.pandas.DataFrame
class Hyperparams(d3m_metadata.hyperparams.Hyperparams):
    # could be used to determine which metafeatures to compute
    pass

class MetafeatureExtractor(FeaturizationTransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = d3m_metadata.metadata.PrimitiveMetadata({
        "primitive_code": {
            "interfaces_version": "2018.1.26"
        },
        "source": {
            "name": "byu-dml",
            "contact": "https://github.com/byu-dml"
        },
        "python_path": "d3m.primitives.d3metafeatureextraction.D3MetafeatureExtraction",
        "version": "v{}".format(__version__),
        "installation": [
            {
                "type": "PIP",
                "package": "d3metafeatureextraction",
                "version": str(__version__)
            }
        ],
        "primitive_family": "METAFEATURE_EXTRACTION",
        "algorithm_types": [
            "DATA_PROFILING",
            "CANONICAL_CORRELATION_ANALYSIS",
            "INFORMATION_ENTROPY",
            "MUTUAL_INFORMATION",
            "SIGNAL_TO_NOISE_RATIO",
            "STATISTICAL_MOMENT_ANALYSIS"
        ],
        "id": "28d12214-8cb0-4ac0-8946-d31fcbcd4142",
        "name": "Dataset Metafeature Extraction"
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not isinstance(inputs, d3m_metadata.container.pandas.DataFrame):
            raise ValueError("inputs must be an instance of 'd3m_metadata.container.pandas.DataFrame'")
        if "target" not in inputs.columns:
            raise ValueError("inputs must contain single-class classification targets in a column labeled 'target'")

        simple_metafeatures = SimpleMetafeatures().compute(inputs)
        statistical_metafeatures = StatisticalMetafeatures().compute(inputs)
        information_thoeretic_metafeatures = InformationTheoreticMetafeatures().compute(inputs)
        landmarking_metafeatures = LandmarkingMetafeatures().compute(inputs)

        metafeatures_df = d3m_metadata.container.pandas.DataFrame(pd.concat([
            simple_metafeatures,
            statistical_metafeatures,
            information_thoeretic_metafeatures,
            landmarking_metafeatures
        ], axis=1))

        return CallResult(metafeatures_df)