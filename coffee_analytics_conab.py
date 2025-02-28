"""
Coffee Analitcs - Companhia Nacional de Abastecimento (CONAB)
Autor: Ronaldo Muinhos
Data: 2025-02-20
Descrição: Análise de dados com Regressão Linear para dados de
produção de café no mundo, utilizando dados da CONAB
Fonte dos Dados: https://www.conab.gov.br/info-agro/safras/serie-historica-das-safras
Tipo: Arábica + Conilon
Instruções Excel: Deixar apenas "Brasil", Remover cabeçalhos e rodapés. Exportar para csv utf-8 (sepadador ";")
FOrmato CSV:    Ano1;Ano2;Ano3;...;AnoN
                Valor1;Valor2;Valor3;...;ValorN

"""
# Importar Bibliotecas de manipulação de dados e viasualização
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Carrega os Arquivos CSV - Converte linha/Coluna e rotula os dados
# Área Plantada (h)
df_area = pd.read_csv("CafeTotalSerieHist_AreaPlantada.csv", sep=";")
df_area = df_area.T.reset_index()
df_area.columns = ["Year", "Value"]
df_area["Year"] = df_area["Year"].astype(int)  # Converter ano para inteiro
df_area["Value"] = df_area["Value"].replace(",", ".", regex=True).astype(float)  # Corrigir ponto decimal e converter para float

# Produtividade (kg/h)
df_yield = pd.read_csv("CafeTotalSerieHist_Produtividade.csv", sep=";") 
df_yield = df_yield.T.reset_index()
df_yield.columns = ["Year", "Value"]
df_yield["Year"] = df_yield["Year"].astype(int)  # Converter ano para inteiro
df_yield["Value"] = df_yield["Value"].replace(",", ".", regex=True).astype(float)  # Corrigir ponto decimal e converter para float

# Produção (x mil sacas)
df_prod = pd.read_csv("CafeTotalSerieHist_Producao.csv", sep=";")  
df_prod = df_prod.T.reset_index()
df_prod.columns = ["Year", "Value"]
df_prod["Year"] = df_prod["Year"].astype(int)  # Converter ano para inteiro
df_prod["Value"] = df_prod["Value"].replace(",", ".", regex=True).astype(float)  # Corrigir ponto decimal e converter para float

# Divide o dataframes de Produção em dois, de acordo com a bianualidade com o objetivo de melhorar a confiabilidade da previsão
df_prod_alta = df_prod[df_prod["Year"] % 2 == 0]  # Anos pares (Produção Alta)
df_prod_baixa = df_prod[df_prod["Year"] % 2 != 0]  # Anos ímpares (Produção Baixa)

# Novas Área (h)
df_newarea = pd.read_csv("CafeTotalSerieHist_AreaFormacao.csv", sep=";")  
df_newarea = df_newarea.T.reset_index()
df_newarea.columns = ["Year", "Value"]
df_newarea["Year"] = df_newarea["Year"].astype(int)  # Converter ano para inteiro
df_newarea["Value"] = df_newarea["Value"].replace(",", ".", regex=True).astype(float)  # Corrigir ponto decimal e converter para float

# Agrupa os dataframes em dataset único
datasets = {"Área Plantada (h)": df_area, "Produtividade (kg/h)": df_yield, "Produção (x mil Sacas)": df_prod, "Produção Biano\Alta (x mil Sacas)": df_prod_alta, "Produção Biano\Baixa (x mil Sacas)": df_prod_baixa, "Novas Áreas (h)": df_newarea}

# Aplicar regressão linear e plotar gráficos com previsão para 5 anos
for element, data in datasets.items():
    grouped = data.groupby("Year")["Value"].sum().reset_index()
    
    X = grouped["Year"].values.reshape(-1, 1)
    y = grouped["Value"].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Criar previsão para 5 anos à frente
    last_year = grouped["Year"].max()
    future_years = np.arange(last_year + 1, last_year + 6).reshape(-1, 1)
    all_years = np.vstack((X, future_years))
    y_pred = model.predict(all_years)
    
    # Separar previsões reais e futuras
    y_pred_future = model.predict(future_years)

    # Calcular o índice de confiabilidade (R²)
    r2_score = model.score(X, y)
    
    # Plotar e exibir gráficos
    plt.figure()
    plt.scatter(X, y, color='blue', label='Dados reais')
    plt.plot(X, model.predict(X), color='red', label='Regressão Linear')
    plt.plot(future_years, y_pred_future, color='green', linestyle='dashed', label='Previsão (5 anos)')
    plt.xlabel("Ano")
    plt.ylabel("Valor")
    plt.title(f"Regressão para {element} - Fonte: Conab (Arábica+Conilon) - Confiabilidade (R²): {r2_score:.4f}")
    plt.legend()
    plt.show()