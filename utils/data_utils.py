import os
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def reset_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_data(csv_path=None):
    if csv_path is None:
        csv_path = "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv"
    data = pd.read_csv(csv_path)
    X = data.drop("fetal_health", axis=1)
    y = data["fetal_health"]
    return X, y


def process_data(X, y, test_size=0.3, seed=42):
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    y = y - 1

    return train_test_split(X_scaled, y, test_size=test_size, random_state=seed)
