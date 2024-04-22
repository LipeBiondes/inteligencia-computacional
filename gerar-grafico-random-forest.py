import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
resultados = pd.read_csv('./resultados/resultados_random_forest.csv')

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(resultados['Previsoes'], label='Previsões')
plt.plot(resultados['Real'], label='Valores Reais')
plt.xlabel('Índice do Dado')
plt.ylabel('Número de Pessoas')
plt.title('Previsões vs Valores Reais')
plt.legend()
plt.grid(True)
plt.show()
