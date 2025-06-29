import pandas as pd
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from train import (
    read_data,
    create_model,
    train_model
)

@pytest.fixture
def sample_data():
    """
    Cria um pequeno dataset simulado para testes unitários.

    Returns:
        pandas.DataFrame: DataFrame com duas colunas de features e uma coluna de target.
    """
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })


def test_read_data():
    """
    Testa se a função read_data carrega corretamente as features e os rótulos.
    """
    X, y = read_data()
    assert not X.empty, "Features retornadas estão vazias."
    assert not y.empty, "Labels retornados estão vazios."
    assert isinstance(X, pd.DataFrame), "X não é um DataFrame."
    assert isinstance(y, pd.Series), "y não é uma Series."


def test_create_model():
    """
    Testa se a função create_model retorna um modelo sequencial válido com camadas treináveis.
    """
    X, _ = read_data()
    model = create_model(X)

    assert isinstance(model, Sequential), "O modelo não é uma instância de Sequential."
    assert model.trainable, "O modelo não está marcado como treinável."
    assert len(model.layers) >= 2, "Modelo deve conter pelo menos 2 camadas."


def test_train_model_success(sample_data):
    """
    Testa se o treinamento do modelo ocorre corretamente com dados simulados.
    """
    X = sample_data.drop('fetal_health', axis=1)
    y = sample_data['fetal_health'] - 1  # categorias devem começar em 0
    y_cat = to_categorical(y)

    model = create_model(X)
    trained_model = train_model(model, X, y_cat, is_train=False)

    assert hasattr(trained_model, "history"), "O modelo treinado não possui histórico."
    assert 'loss' in trained_model.history.history, "Histórico não contém 'loss'."
    assert 'val_loss' in trained_model.history.history, "Histórico não contém 'val_loss'."

    last_loss = trained_model.history.history['loss'][-1]
    val_loss = trained_model.history.history['val_loss'][-1]

    assert last_loss > 0, "O valor de loss final não é positivo."
    assert val_loss > 0, "O valor de val_loss final não é positivo."


def test_create_model_with_invalid_input():
    """
    Testa a criação de modelo com entrada inválida.
    """
    with pytest.raises(Exception):
        create_model(None)


def test_train_model_with_invalid_data():
    """
    Testa o comportamento do modelo quando dados inválidos são fornecidos.
    """
    model = create_model(pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))
    
    with pytest.raises(Exception):
        train_model(model, None, None, is_train=False)
