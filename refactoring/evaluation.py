from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error, classification_report, accuracy_score


def evaluate_model(model: RandomForestClassifier, X_test, y_test):
    y_pred_rf = model.predict(X_test)
    mse_rf = root_mean_squared_error(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    print(f"Error Cuadrático Medio (MSE) - Random Forest: {round(mse_rf, 4)}")
    print(f"Accuracy - Random Forest: {round(accuracy_rf, 4)}")
    print("Informe de clasificación - Random Forest:")
    print(report_rf)
