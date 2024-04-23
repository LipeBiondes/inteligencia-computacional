import pandas as pd
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

# Carregar os dados de treinamento
dados_treino = pd.read_csv('./planilhas/acidentes_por_ano_mes_dia_treino.csv')

# Separar as features (X) e o target (y) para treinamento
X_treino = dados_treino[['ano', 'mes', 'dia']]
y_treino = dados_treino['Acidentes']

# Instanciar e treinar o modelo ElasticNet
modelo = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Defina os parâmetros alpha e l1_ratio conforme necessário
modelo.fit(X_treino, y_treino)

# Fazer previsões para os dados de teste (ano, mês, dia)
dados_teste = pd.read_csv('./planilhas/acidentes_por_ano_mes_dia_teste.csv')
X_teste = dados_teste[['ano', 'mes', 'dia']]
previsoes = modelo.predict(X_teste)

# Adicionar as previsões ao DataFrame de teste
dados_teste['Previsoes'] = previsoes

# Agrupar os dados reais por mês
dados_reais_agrupados = dados_teste.groupby('mes')['Acidentes'].sum().reset_index()

# Agrupar as previsões por mês
previsoes_agrupadas = dados_teste.groupby('mes')['Previsoes'].sum().reset_index()

# Plotar o gráfico com os resultados
plt.plot(dados_reais_agrupados['mes'], dados_reais_agrupados['Acidentes'], label='Dados Reais', marker='o')
plt.plot(previsoes_agrupadas['mes'], previsoes_agrupadas['Previsoes'], label='Previsões', marker='x')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')
plt.title('Comparação entre Dados Reais e Previsões de Número de Acidentes por Mês (2023)')
plt.legend()
plt.show()
