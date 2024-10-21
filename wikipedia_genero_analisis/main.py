from fastapi import FastAPI
from modeling.schemas import TrainingData
from modeling.model_training import train_and_log_model
from modeling import preprocess, mlflow_helper, evaluation
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("wikipedia")
mlflow.sklearn.autolog(log_post_training_metrics=False)
app = FastAPI()

@app.post("/train")
def train_model(data: TrainingData):

    columns = ["E_NEds", "E_Bpag", "firstDay", "lastDay", "NEds", "NDays", "NActDays", "NPages", "NPcreated", "pagesWomen", "wikiprojWomen", "ns_user", "ns_wikipedia", "ns_talk", "ns_userTalk", "ns_content", "weightIJ", "NIJ"]
    # Convert lists to DataFrames/Series
    X_train, X_test, y_train, y_test = preprocess.get_data(columns)

    try:
        train_and_log_model(
            X_train, y_train, X_test, y_test,
            classifier_type=data.classifier_type,
            hyperparameters=data.hyperparameters
        )
        return {"status": "Model trained and logged successfully"}
    except Exception as e:
        return {"error": str(e)}