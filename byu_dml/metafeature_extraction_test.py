import pandas as pd

from d3m.container.pandas import DataFrame

from metafeature_extraction import MetafeatureExtractor

if __name__ == '__main__':
    infile_path = "data/learningData.csv"
    df = DataFrame(pd.read_csv(infile_path))
    df = df.rename(columns={"Class": "target"})
    df.drop("d3mIndex", axis=1, inplace=True)
    metafeatures = MetafeatureExtractor(hyperparams=None).produce(inputs=df).value
    print(metafeatures)
