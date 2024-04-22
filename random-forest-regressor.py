import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Carregar os dados
dados_2020_2022 = pd.read_csv('./acidentes_2020-2022-ano-mes-dia.csv', sep=',', encoding='utf-8')
dados_2023 = pd.read_csv('./acidentes_2023-ano-mes-dia.csv', sep=',', encoding='utf-8')

# Dividir os dados de treino em features (X) e target (y)
X_treino = dados_2020_2022.drop(['ano', 'pessoas'], axis=1)
y_treino = dados_2020_2022['pessoas']

# Dividir os dados de teste em features (X) e target (y)
X_teste = dados_2023.drop(['ano', 'pessoas'], axis=1)
y_teste = dados_2023['pessoas']

# Instanciar e treinar o modelo (Random Forest Regressor, por exemplo)
modelo = RandomForestRegressor()
modelo.fit(X_treino, y_treino)

# Fazer previsões para o ano de 2023
previsoes = modelo.predict(X_teste)

# Avaliar o modelo
mse = mean_squared_error(y_teste, previsoes)
print(f"MSE: {mse}")

# Criar DataFrame com as previsões e o MSE
resultados = pd.DataFrame({'Previsoes': previsoes, 'Real': y_teste})
resultados['Erro'] = resultados['Previsoes'] - resultados['Real']
resultados['Erro Quadrático'] = resultados['Erro'] ** 2
resultados.to_csv('resultados_random_forest.csv', index=False)
