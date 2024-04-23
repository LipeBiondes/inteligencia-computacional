import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Carregar os dados de treinamento e teste
dados_treino = pd.read_csv('./planilhas/acidentes_por_ano_mes_treino.csv')
dados_teste = pd.read_csv('./planilhas/acidentes_por_ano_mes_teste.csv')

# Separar as features (ano e mês) e o target (número de acidentes) para treinamento
X_treino = dados_treino[['ano', 'mes']]
y_treino = dados_treino['Acidentes']

# Instanciar e treinar o modelo K-NN
modelo = KNeighborsRegressor(n_neighbors=5)  # Define o número de vizinhos desejado (n_neighbors)
modelo.fit(X_treino, y_treino)

# Fazer previsões para os dados de teste
X_teste = dados_teste[['ano', 'mes']]
previsoes = modelo.predict(X_teste)

# Plotar o gráfico com os dados de teste e as previsões
plt.plot(dados_teste['mes'], dados_teste['Acidentes'], color='blue', marker='o', label='Dados Reais')
plt.plot(dados_teste['mes'], previsoes, color='red', marker='o', linestyle='dashed', label='Previsões')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')
plt.title('Previsões de Acidentes por Mês')
plt.legend()
plt.grid(True)
plt.show()
