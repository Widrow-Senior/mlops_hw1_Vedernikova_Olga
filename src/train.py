from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import yaml
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Classification')

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")

    with open('C:\\Users\\Legion\\Documents\\MLOps\\HW1\\params.yaml') as file:
        params = yaml.safe_load(file)

    train_df = pd.read_csv("C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\processed\\train.csv")
    test_df = pd.read_csv("C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\processed\\test.csv")

    X_train, y_train = train_df.drop('y', axis=1).values, train_df['y']
    X_test, y_test = test_df.drop('y', axis=1).values, test_df['y']

    model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    mlflow.log_artifact("model.pkl")
