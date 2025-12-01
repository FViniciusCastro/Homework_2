from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# ============================================================
#                OLS COM SKLEARN (PRONTO)
# ============================================================

# Criar o modelo
ols_sklearn = LinearRegression()

# Treinar o modelo com X_train e y_train
ols_sklearn.fit(X_train, y_train)

# Prever no treino e teste
y_pred_train_sk = ols_sklearn.predict(X_train)
y_pred_test_sk = ols_sklearn.predict(X_test)

# Calcular RMSE
rmse_train_sk = np.sqrt(mean_squared_error(y_train, y_pred_train_sk))
rmse_test_sk = np.sqrt(mean_squared_error(y_test, y_pred_test_sk))

# Calcular R²
r2_train_sk = r2_score(y_train, y_pred_train_sk)
r2_test_sk = r2_score(y_test, y_pred_test_sk)

# Mostrar resultados
print("=========== RESULTADOS OLS (SKLEARN) ===========")
print("RMSE Treino:", rmse_train_sk)
print("RMSE Teste :", rmse_test_sk)
print("R² Treino  :", r2_train_sk)
print("R² Teste   :", r2_test_sk)
print("===============================================")
