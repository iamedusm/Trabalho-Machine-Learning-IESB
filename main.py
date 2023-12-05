import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

##### Carregar os arquivos CSV
df_dolar = pd.read_csv('./dados/BRL=X.csv').rename(columns={'Close': 'Dolar_Close'})
df_bvsp = pd.read_csv('./dados/^BVSP.csv').rename(columns={'Close': 'BVSP_Close'})
df_petroleo = pd.read_csv('./dados/Petroleo_Tratado.csv').rename(columns={'Data': 'Date', 'Ultimo': 'Petroleo_Close'})
df_ouro = pd.read_csv('./dados/Ouro_Tratado.csv').rename(columns={'Data': 'Date', 'Ultimo': 'Ouro_Close'})
df_gol = pd.read_csv('./dados/GOLL4_SA.csv').rename(columns={'Close': 'GOL_Close'})

##### Mesclar os dataframes com base na coluna 'Date'
df_merged = df_gol.merge(df_dolar, on='Date', how='left')
df_merged = df_merged.merge(df_bvsp, on='Date', how='left')
df_merged = df_merged.merge(df_petroleo, on='Date', how='left')
df_merged = df_merged.merge(df_ouro, on='Date', how='left')

# Remover linhas com valores NaN e preparar X e y
df_merged.dropna(inplace=True)
X = df_merged[['Dolar_Close', 'BVSP_Close', 'Petroleo_Close', 'Ouro_Close']]
y = df_merged['GOL_Close']

# Treinar o modelo de regressão linear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calcular e imprimir métricas de desempenho
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')