import data_loading
import evaluation
import preprocess
import train


_RANDOM_STATE = 42
raw_data = data_loading.load_data("notebooks/1.EDA_Gender_Gap_in_Spanish_WP/data/data.csv")
main_columns = ['NEds', 'NActDays', 'pagesWomen', 'wikiprojWomen']
transformed_data = preprocess.preprocess_data(raw_data, main_columns)
X_train, X_test, y_train, y_test = preprocess.split_data(transformed_data, main_columns, "gender",
                                                         0.3, _RANDOM_STATE)
model = train.train_model(X_train, y_train, _RANDOM_STATE)
evaluation.evaluate_model(model, X_test, y_test)
