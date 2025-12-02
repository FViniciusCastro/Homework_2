#%%

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Imports OK")

#%%

df_train = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv")
df_test = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv")

cols_to_drop = ["class", "plate", "MJD"]


X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
y_train = df_train["redshift"]

X_test = df_test.drop(columns=cols_to_drop + ["redshift"])
y_test = df_test["redshift"]

print("Dados carregados e pré-processados.")
#%%

ols_sklearn = LinearRegression()
ols_sklearn.fit(X_train, y_train)

y_pred_train_sk = ols_sklearn.predict(X_train)
y_pred_test_sk = ols_sklearn.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_sk))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_sk))

r2_train = r2_score(y_train, y_pred_train_sk)
r2_test = r2_score(y_test, y_pred_test_sk)

print("\n=========== RESULTADOS OLS (SKLEARN) ===========")
print(f"RMSE Treino: {rmse_train}")
print(f"RMSE Teste : {rmse_test}")
print(f"R² Treino : {r2_train}")
print(f"R² Teste : {r2_test}")
print("===============================================")

#%% OLS DO ZERO — FUNÇÕES E EXECUÇÃO
def add_intercept(X):
    """Adiciona a coluna de intercepto (1s) à matriz X."""
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def compute_beta_ols(X, y):
    """Calcula os coeficientes beta usando a solução fechada do OLS."""
    XT = X.T
    return np.linalg.inv(XT @ X) @ (XT @ y)

def predict(X, beta):
    """Gera previsões usando o modelo linear."""
    return X @ beta

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_manual(y_true, y_pred):
    """R² (coeficiente de determinação) calculado manualmente."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def ols_from_scratch(X_train, y_train, X_test, y_test):
    """
    Executa o pipeline OLS manual:
    - converte para numpy
    - adiciona intercepto
    - calcula beta via (X^T X)^{-1} X^T y
    - retorna coeficientes e métricas
    """
    # Garantir arrays numpy e formatos corretos
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test  = np.array(y_test).reshape(-1, 1)

    # Adiciona intercepto
    X_train_i = add_intercept(X_train)
    X_test_i  = add_intercept(X_test)

    # Calcula beta (atenção: pode lançar erro se X^T X singular)
    beta = compute_beta_ols(X_train_i, y_train)

    # Previsões
    y_pred_train = predict(X_train_i, beta)
    y_pred_test  = predict(X_test_i, beta)

    # Métricas
    results = {
        "beta": beta,
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_r2": r2_score_manual(y_train, y_pred_train),
        "test_r2": r2_score_manual(y_test, y_pred_test)
    }
    return results

# === Executar OLS do zero ===
results_ols = ols_from_scratch(X_train, y_train, X_test, y_test)

print("\n====== RESULTADOS DO OLS (DO ZERO) ======")
print("Coeficientes (beta):")
print(results_ols["beta"].flatten())   # mostra em 1D
print("\n--- MÉTRICAS ---")
print(f"RMSE Treino: {results_ols['train_rmse']:.6f}")
print(f"RMSE Teste : {results_ols['test_rmse']:.6f}")
print(f"R² Treino  : {results_ols['train_r2']:.6f}")
print(f"R² Teste   : {results_ols['test_r2']:.6f}")
print("=========================================")

#%%

print("\n========== COMPARAÇÃO DIRETA ==========")
print(f"RMSE Teste (do zero) : {results_ols['test_rmse']}")
print(f"RMSE Teste (sklearn) : {rmse_test}")
print(f"R² Teste (do zero) : {results_ols['test_r2']}")
print(f"R² Teste (sklearn) : {r2_test}")
print("=======================================")

