import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Carregar os dados
dados = pd.read_csv('acidentes_por_ano.csv', sep=';', encoding='utf-8')

# Dividir os dados em treino (2020-2022) e teste (2023)
dados_treino = dados[dados['ano'].isin([2020, 2021, 2022])]
dados_teste = dados[dados['ano'] == 2023]

# Dividir os dados de treino em features (X) e target (y)
X_treino = dados_treino.drop(['ano', 'pessoas'], axis=1)
y_treino = dados_treino['pessoas']

# Dividir os dados de teste em features (X) e target (y)
X_teste = dados_teste.drop(['ano', 'pessoas'], axis=1)
y_teste = dados_teste['pessoas']

# Instanciar e treinar o modelo (Random Forest Regressor, por exemplo)
modelo = RandomForestRegressor()
modelo.fit(X_treino, y_treino)

# Fazer previsões para o ano de 2023
previsoes = modelo.predict(X_teste)

# Avaliar o modelo
mse = mean_squared_error(y_teste, previsoes)
print(f"MSE: {mse}")
