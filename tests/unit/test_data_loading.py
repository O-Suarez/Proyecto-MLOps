from pandas import DataFrame

from refactoring import data_loading


def test_load_wikipedia():
    data_path = "notebooks/1.EDA_Gender_Gap_in_Spanish_WP/data/data.csv"
    wikipedia_df = data_loading.load_wikipedia(data_path)
    assert type(wikipedia_df) is DataFrame
