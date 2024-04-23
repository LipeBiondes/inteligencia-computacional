import pandas as pd

# Carregar os dados
dados = pd.read_csv('./planilhas/acidentes_2020-2022-ano-mes-dia.csv', sep=',', encoding='utf-8')

# Agrupar os dados por ano e mês e somar o número de acidentes em cada mês
acidentes_por_ano_mes = dados.groupby(['ano', 'mes']).size().reset_index(name='Acidentes')

# Exibir os dados agrupados
print(acidentes_por_ano_mes)

# Salvar os dados agrupados em um arquivo CSV
acidentes_por_ano_mes.to_csv('./planilhas/acidentes_por_ano_mes_treino.csv', index=False)

# Carregar os dados de teste
dados_teste = pd.read_csv('./planilhas/acidentes_2023-ano-mes-dia.csv', sep=',', encoding='utf-8')

# Agrupar os dados de teste por ano e mês e somar o número de acidentes em cada mês
acidentes_por_ano_mes_teste = dados_teste.groupby(['ano', 'mes']).size().reset_index(name='Acidentes')

# Exibir os dados agrupados de teste
print(acidentes_por_ano_mes_teste)

# Salvar os dados agrupados de teste em um arquivo CSV
acidentes_por_ano_mes_teste.to_csv('./planilhas/acidentes_por_ano_mes_teste.csv', index=False)
