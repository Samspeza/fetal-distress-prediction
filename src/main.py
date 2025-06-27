from test.train import (
    read_data,
    process_data,
    create_model,
    train_model,
    evaluate_model
)

from mlflow_utils import config_mlflow


def main():
    # Setup
    config_mlflow()

    # Pipeline
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X_train)
    model, _ = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
