from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, random_state: int) -> RandomForestClassifier:
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model
