import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Carregar os dados de treinamento
dados_treino = pd.read_csv('./planilhas/acidentes_por_ano_mes_dia_treino.csv')

# Separar as features (X) e o target (y) para treinamento
X_treino = dados_treino[['ano', 'mes', 'dia']]
y_treino = dados_treino['Acidentes']

# Instanciar e treinar o modelo XGBoost
modelo = XGBRegressor()
modelo.fit(X_treino, y_treino)

# Fazer previsões para os dados de teste (ano, mês, dia)
dados_teste = pd.read_csv('./planilhas/acidentes_por_ano_mes_dia_teste.csv')
X_teste = dados_teste[['ano', 'mes', 'dia']]
previsoes = modelo.predict(X_teste)

# Adicionar as previsões ao DataFrame de teste
dados_teste['Previsoes'] = previsoes

# Calcular os acertos para cada mês
acertos_por_mes = []
for mes in dados_teste['mes'].unique():
    dados_mes = dados_teste[dados_teste['mes'] == mes]
    precisao_mes = 100 * (1 - abs(dados_mes['Acidentes'] - dados_mes['Previsoes']) / dados_mes['Acidentes'])
    media_mes = precisao_mes.mean()
    acertos_por_mes.append(media_mes)

# Plotar o gráfico com a média de acertos por mês
meses = dados_teste['mes'].unique()
plt.plot(meses, acertos_por_mes, label='Média de Acertos', marker='o')
plt.xlabel('Mês')
plt.ylabel('Porcentagem de Acertos')
plt.title('Média de Acertos das Previsões de Número de Acidentes por Mês (2023) - XGBoost')
plt.legend()
plt.show()

# Calcular a média de acertos
media_acertos = sum(acertos_por_mes) / len(acertos_por_mes)
print(f'Média de acertos por mês: {media_acertos:.2f}%')
