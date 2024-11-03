import pandas as pd
from pandas import DataFrame


def load_wikipedia(data_path: str) -> DataFrame:
    df = pd.read_csv(data_path)
    return df
