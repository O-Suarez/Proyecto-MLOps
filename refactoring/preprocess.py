import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import model_selection
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from refactoring import data_loading

transformer = ColumnTransformer([
    ('num', MinMaxScaler(), make_column_selector(dtype_include=np.number)),
    ('cat', OneHotEncoder(), make_column_selector(dtype_include=object))
])

def remove_outliers(raw_data: DataFrame, columns: list[str]):
    z_scores = np.abs(stats.zscore(raw_data[columns]))
    filtered_data = raw_data[(z_scores < 3).all(axis=1)]
    return filtered_data

def remove_missing_values(raw_data: DataFrame):
    return raw_data.dropna()

def get_feature_names(column_transformer, input_features):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if name != 'remainder':
            # Skip transformers with no columns
            if len(columns) == 0:
                continue
            if isinstance(transformer, Pipeline):
                transformer = transformer.steps[-1][1]
            try:
                if hasattr(transformer, 'get_feature_names_out'):
                    if isinstance(transformer, OneHotEncoder):
                        names = transformer.get_feature_names_out(columns)
                    else:
                        names = transformer.get_feature_names_out()
                else:
                    names = columns  # Use original column names if method not available
            except NotFittedError:
                continue  # Skip transformer if not fitted
            feature_names.extend(names)
    return feature_names

def get_data(main_columns=['NEds', 'NActDays', 'pagesWomen', 'wikiprojWomen'],
             raw_data="notebooks/1.EDA_Gender_Gap_in_Spanish_WP/data/data.csv"):
    raw_data = data_loading.load_wikipedia(raw_data)
    transformed_data = remove_missing_values(raw_data)

    numeric_columns = transformed_data.select_dtypes(include=[np.number]).columns
    transformed_data = remove_outliers(transformed_data, numeric_columns)
    
    X = transformed_data[main_columns].copy()
    y = transformed_data["gender"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    # Pipeline for applying transformations and SMOTE to the training data
    training_pipeline = ImbPipeline([
        ('transform', transformer),
        ('smote', SMOTE(random_state=42))
    ])

    # Fit the pipeline and apply transformations and SMOTE to training data
    X_train_prepared, y_train_prepared = training_pipeline.fit_resample(X_train, y_train)

    # Apply only transformations to test data
    transformation_pipeline = Pipeline([
        ('transform', transformer)
    ])
    # Fit transformer on training data and transform test data
    transformation_pipeline.fit(X_train)
    X_test_prepared = transformation_pipeline.transform(X_test)

    # Get feature names after transformation
    feature_names = get_feature_names(transformer, X_train.columns)

    # Convert X_train_prepared and X_test_prepared to DataFrames with feature names
    X_train_prepared = pd.DataFrame(X_train_prepared, columns=feature_names)
    X_test_prepared = pd.DataFrame(X_test_prepared, columns=feature_names)

    return X_train_prepared, X_test_prepared, y_train_prepared, y_test