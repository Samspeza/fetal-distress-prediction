# 🧠 Detecção de Sofrimento Fetal por Cardiotocografia

> Construção de um pipeline preditivo para classificação clínica com base em dados de cardiotocografia.

## 🩺 Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de aprendizado de máquina capaz de classificar automaticamente exames de **cardiotocografia** em três categorias clínicas:

- **Normal**: Parto dentro da normalidade.
- **Suspeito**: Sinais de possível risco fetal.
- **Doente**: Indícios de sofrimento fetal iminente (recomendada intervenção médica).

A proposta busca aplicar as melhores práticas em ciência de dados e machine learning na criação de um **pipeline completo**, desde a análise exploratória até a avaliação de desempenho do modelo.

---

## 📊 Dataset

Utilizamos um conjunto de dados real contendo leituras extraídas de exames de cardiotocografia, com as seguintes colunas:

- **FHR** (Fetal Heart Rate)
- **UC** (Uterine Contractions)
- **STV**, **LTV**, entre outros indicadores técnicos...
- **Classe alvo**: {1 = Normal, 2 = Suspeito, 3 = Doente}

Fonte: [Coloque aqui o link do dataset, se público]

---

## ⚙️ Pipeline de Desenvolvimento

1. **Pré-processamento de Dados**
   - Limpeza e normalização
   - Tratamento de valores ausentes
2. **Análise Exploratória**
   - Visualizações
   - Correlações e insights clínicos
3. **Balanceamento**
   - Técnicas como SMOTE ou undersampling
4. **Treinamento de Modelos**
   - Avaliação de diferentes algoritmos (Random Forest, SVM, XGBoost etc.)
5. **Validação**
   - Cross-validation e métricas (accuracy, recall, F1-score)
6. **Exportação do Modelo**
   - `joblib` ou `pickle`

---

## 📈 Resultados Esperados

O modelo final deverá fornecer:
- Classificação em tempo real de novos exames
- Acurácia e sensibilidade adequadas à aplicação clínica
- Interpretação dos fatores mais importantes na decisão do modelo

---

## 🧰 Tecnologias Utilizadas

- Python 3.10+
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebooks

---

## 👩‍⚕️ Aplicações Futuras

Este projeto pode servir como base para:
- Sistemas de apoio à decisão médica (CDSS)
- Integração com prontuários eletrônicos (EHR)
- Aplicações embarcadas em clínicas obstétricas

---

## 🚀 Como Executar

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/fetal-distress-prediction

# Acesse a pasta
cd fetal-distress-prediction

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook principal
jupyter notebook
