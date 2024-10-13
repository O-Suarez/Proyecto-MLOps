from imblearn.over_sampling import SMOTE
import numpy as np
from pandas import DataFrame
from scipy import stats
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler



def preprocess_data(raw_data: DataFrame, columns: list[str]):
    z_scores = np.abs(stats.zscore(raw_data[columns]))
    raw_data = raw_data[(z_scores < 3).all(axis=1)]
    raw_data["gender"] = raw_data["gender"].astype("category")
    return raw_data

def split_data(
        data: DataFrame,
        x_columns: list[str],
        y_column: str,
        test_size: float,
        random_state: int):
    X = data[x_columns]
    y = data[y_column]
    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled, y_resampled,
                                                                        test_size=test_size,
                                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test
