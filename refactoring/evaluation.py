from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, classification_report, accuracy_score


def evaluate_model(model, X_test, y_test, mlflow_run):
    y_pred_rf = model.predict(X_test)
    mse_rf = root_mean_squared_error(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    model_name = mlflow_run.data.tags["estimator_name"]

    print(f"Error Cuadrático Medio (MSE) - {model_name}: {round(mse_rf, 4)}")
    print(f"Accuracy - {model_name}: {round(accuracy_rf, 4)}")
    print(f"Informe de clasificación - {model_name}:")
    print(report_rf)
