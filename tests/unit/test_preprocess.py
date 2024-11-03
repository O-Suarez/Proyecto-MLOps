from refactoring import data_loading, preprocess


def test_remove_outliers():
    main_columns = ['NEds', 'NActDays', 'pagesWomen', 'wikiprojWomen']
    raw_data = data_loading.load_wikipedia("notebooks/1.EDA_Gender_Gap_in_Spanish_WP/data/data.csv")
    transformed_data = preprocess.remove_outliers(raw_data, main_columns)
    raw_data_std = raw_data[main_columns].std()
    transformed_data_std = transformed_data[main_columns].std()
    assert (transformed_data_std < raw_data_std).all()
