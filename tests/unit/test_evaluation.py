import mlflow
from sklearn import model_selection

from refactoring import evaluation, data_loading, preprocess


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wikipedia")


def test_evaluate_model():
    main_columns = ['NEds', 'NActDays', 'pagesWomen', 'wikiprojWomen']
    raw_data = data_loading.load_wikipedia("notebooks/1.EDA_Gender_Gap_in_Spanish_WP/data/data.csv")
    transformed_data = preprocess.remove_outliers(raw_data, main_columns)
    _, X_test, _, y_test = model_selection.train_test_split(transformed_data[main_columns],
                                                            transformed_data["gender"],
                                                            test_size=0.3)
    model_uri = f"runs:/09e5ebd61aa14e42a900adf9bc2c7dd4/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    evaluation.evaluate_model(loaded_model, X_test, y_test)
