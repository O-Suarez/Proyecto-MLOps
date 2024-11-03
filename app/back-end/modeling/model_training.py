# model_training.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.svm import SVC
import xgboost as xgb

def train_and_log_model(X_train, y_train, X_test, y_test, classifier_type='logistic_regression', hyperparameters={}, random_state=42):
    mlflow.sklearn.autolog()
    mlflow.autolog()
    with mlflow.start_run() as run:
        if classifier_type == 'logistic_regression':
            clf = LogisticRegression(random_state=random_state, **hyperparameters)
        elif classifier_type == 'random_forest':
            clf = RandomForestClassifier(random_state=random_state, **hyperparameters)
        elif classifier_type == 'svm':
            clf = SVC(random_state=random_state, **hyperparameters)
        elif classifier_type == 'xgboost':
            clf = xgb.XGBClassifier(random_state=random_state, **hyperparameters)
        else:
            raise ValueError("Unsupported classifier type")

        model_pipeline = Pipeline(steps=[
            ('classifier', clf)
        ])

        model_pipeline.fit(X_train, y_train)
        accuracy = model_pipeline.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(hyperparameters)
        mlflow.log_params({"classifier_type": classifier_type})
        mlflow.log_params({"random_state": random_state})
        # Guarda el modelo en MLFlow
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            signature=infer_signature(X_train, model_pipeline.predict(X_train)),
            input_example=X_train.head(1)
        )
        return accuracy