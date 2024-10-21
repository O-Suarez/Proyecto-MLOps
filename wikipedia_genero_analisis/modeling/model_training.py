# model_training.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models.signature import infer_signature

def train_and_log_model(X_train, y_train, X_test, y_test, classifier_type='logistic_regression', hyperparameters={}):
    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        if classifier_type == 'logistic_regression':
            clf = LogisticRegression(random_state=42, **hyperparameters)
        elif classifier_type == 'random_forest':
            clf = RandomForestClassifier(random_state=42, **hyperparameters)
        else:
            raise ValueError("Unsupported classifier type")

        model_pipeline = Pipeline(steps=[
            ('classifier', clf)
        ])

        model_pipeline.fit(X_train, y_train)
        accuracy = model_pipeline.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            signature=infer_signature(X_train, model_pipeline.predict(X_train)),
            input_example=X_train.head(1)
        )