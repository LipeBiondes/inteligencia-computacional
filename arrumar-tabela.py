import pandas as pd

# Carregar os dados
dados = pd.read_csv('./planilhas/acidentes_2020-2022-ano-mes-dia.csv', sep=',', encoding='utf-8')

# Agrupar os dados por ano, mês, dia e contar o número de acidentes em cada combinação
acidentes_por_ano_mes_dia = dados.groupby(['ano', 'mes', 'dia']).size().reset_index(name='Acidentes')

# Exibir os dados agrupados
print(acidentes_por_ano_mes_dia)

# Salvar os dados agrupados em um arquivo CSV
acidentes_por_ano_mes_dia.to_csv('./planilhas/acidentes_por_ano_mes_dia_treino.csv', index=False)

# Carregar os dados de teste
dados_teste = pd.read_csv('./planilhas/acidentes_2023-ano-mes-dia.csv', sep=',', encoding='utf-8')

# Agrupar os dados de teste por ano, mês, dia e contar o número de acidentes em cada combinação
acidentes_por_ano_mes_dia_teste = dados_teste.groupby(['ano', 'mes', 'dia']).size().reset_index(name='Acidentes')

# Exibir os dados agrupados de teste
print(acidentes_por_ano_mes_dia_teste)

# Salvar os dados agrupados de teste em um arquivo CSV
acidentes_por_ano_mes_dia_teste.to_csv('./planilhas/acidentes_por_ano_mes_dia_teste.csv', index=False)
