import pandas as pd
import numpy as np
import os

from d3m.container.pandas import DataFrame
from d3m.container.dataset import Dataset
from load_d3m_dataset import load_dataset
from imputer import RandomSamplingImputer

def print_missing_vals_info(df, df_name):
    num_empty_cols = 0
    total_num_nan = 0
    for feature in df:
        num_nan = np.sum(df[feature].isnull())
        if num_nan == len(df[feature]):
            num_empty_cols += 1
        else:
            total_num_nan += num_nan
    print('\nMissing Values for ', df_name)
    print('\nEmpty columns: ', num_empty_cols)
    print('Total missing values (not counting empty columns): ', total_num_nan)

if __name__ == '__main__':
    df = load_dataset()
    df.replace("", np.nan, inplace=True)
    print_missing_vals_info(df, 'Input Dataset')
    imputer = RandomSamplingImputer(hyperparams=None, random_seed=0)
    imputer.set_training_data(inputs=df)
    imputer.fit()
    new_df = imputer.produce(inputs=df).value
    print_missing_vals_info(new_df, 'Imputed Dataset')