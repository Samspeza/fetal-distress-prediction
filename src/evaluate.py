import mlflow

def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)