import os
import random

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, InputLayer
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def reset_seeds():
    """
    Reset the seeds for random number generators.

    This function sets the seeds for the `os`, `tf.random`, `np.random`, and `random`
    modules to ensure reproducibility in random number generations.

    Parameters:
        None

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data():
    """
    Reads the data from a CSV file and returns the feature matrix X and target vector y.

    Returns:
        X (pandas.DataFrame): The feature matrix of shape (n_samples, n_features).
        y (pandas.Series): The target vector of shape (n_samples,).
    """
    data = pd.read_csv(
        'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv')
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y


def process_data(X, y):
    """
    Preprocesses the data by standardizing the feature values and splitting the
    data into training and testing sets.

    Parameters:
        X (pandas.DataFrame): The input data containing the features.
        y (pandas.Series): The target variable.

    Returns:
        X_train (pandas.DataFrame): The preprocessed training data.
        X_test (pandas.DataFrame): The preprocessed testing data.
        y_train (pandas.Series): The training labels.
        y_test (pandas.Series): The testing labels.
    """
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    y_train = y_train - 1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test


def create_model(X):
    """
    Creates a neural network model for classification based on the given input data.

    Parameters:
        X (numpy.ndarray): The input data array. It should have a shape of (num_samples,
         num_features).

    Returns:
        tensorflow.keras.models.Sequential: The created neural network model.
    """
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1],)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def config_mlflow():
    """
    Configures the MLflow settings for tracking experiments.

    Sets the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment
     variables to provide authentication for accessing the MLflow tracking server.

    Sets the MLflow tracking URI to 'https://dagshub.com/renansantosmendes/mlops-ead.mlflow'
    to specify the location where the experiment data will be logged.

    Enables autologging of TensorFlow models by calling `mlflow.tensorflow.autolog()`.
    This will automatically log the TensorFlow models, input examples, and model signatures
    during training.

    Parameters:
        None

    Returns:
        None
    """
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/mlops-ead-2025.mlflow')

    mlflow.keras.autolog(log_models=True,
                         log_input_examples=True,
                         log_model_signatures=True)


def train_model(model, X_train, y_train, is_train=True):
    """
    Train a machine learning model using the provided data.

    Parameters:
    - model: The machine learning model to train.
    - X_train: The training data.
    - y_train: The target labels.
    - is_train: (optional) Flag indicating whether to register the
    model with mlflow.
                Defaults to True.

    Returns:
    None
    """
    with mlflow.start_run(run_name='experiment_mlops_ead') as run:
        model.fit(X_train,
                  y_train,
                  epochs=50,
                  validation_split=0.2,
                  verbose=3)
    if is_train:
        run_uri = f'runs:/{run.info.run_id}'
        mlflow.register_model(run_uri, 'fetal_health')

def reset_seeds():
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data(csv_path=None):
    if csv_path is None:
        csv_path = "https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv"
    data = pd.read_csv(csv_path)
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]
    return X, y


def process_data(X, y):
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    y_train -= 1
    y_test -= 1

    return X_train, X_test, y_train, y_test


def create_model(X):
    reset_seeds()
    model = Sequential([
        InputLayer(input_shape=(X.shape[1],)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss')
    ]


def train_model(model, X_train, y_train, is_train=True):
    import mlflow  # evitar import circular com mlflow_utils

    with mlflow.start_run(run_name="experiment_mlops_ead") as run:
        model.fit(
            X_train,
            y_train,
            epochs=50,
            validation_split=0.2,
            callbacks=get_callbacks(),
            verbose=2
        )

        if is_train:
            from mlflow_utils import register_model
            register_model(run.info.run_id)

        return model, run


def evaluate_model(model, X_test, y_test):
    import mlflow
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)



if __name__ == "__main__":
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)
