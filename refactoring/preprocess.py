import numpy as np
from pandas import DataFrame
from scipy import stats


def remove_outliers(raw_data: DataFrame, columns: list[str]):
    z_scores = np.abs(stats.zscore(raw_data[columns]))
    raw_data = raw_data[(z_scores < 3).all(axis=1)]
    return raw_data
