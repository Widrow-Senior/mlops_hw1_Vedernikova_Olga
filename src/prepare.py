import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import yaml

with open('C:\\Users\\Legion\\Documents\\MLOps\\HW1\\params.yaml') as file:
    params = yaml.safe_load(file)

path = "C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\raw\\data.csv"

df = pd.read_csv(path)

le = LabelEncoder()
y = le.fit_transform(df['Species'])

X = df.drop('Species', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['split_ratio'], random_state=params['random_state'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\processed', exist_ok=True)

X_train_df = pd.DataFrame(X_train_scaled)
X_train_df['y'] = y_train
X_train_df.to_csv('C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\processed\\train.csv', index=False)

X_test_df = pd.DataFrame(X_test_scaled)
X_test_df['y'] = y_test
X_test_df.to_csv('C:\\Users\\Legion\\Documents\\MLOps\\HW1\\data\\processed\\test.csv', index=False)