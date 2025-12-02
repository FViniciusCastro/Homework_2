#%%

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Imports OK")


df_train = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv")
df_test = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv")

cols_to_drop = ["class", "plate", "MJD"]


X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
y_train = df_train["redshift"]

X_test = df_test.drop(columns=cols_to_drop + ["redshift"])
y_test = df_test["redshift"]

print("Dados carregados e pré-processados.")
#%% OLS COM SKLEARN

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
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test  = np.array(y_test).reshape(-1, 1)

    X_train_i = add_intercept(X_train)
    X_test_i  = add_intercept(X_test)

    beta = compute_beta_ols(X_train_i, y_train)

    y_pred_train = predict(X_train_i, beta)
    y_pred_test  = predict(X_test_i, beta)

    results = {
        "beta": beta,
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_r2": r2_score_manual(y_train, y_pred_train),
        "test_r2": r2_score_manual(y_test, y_pred_test)
    }
    return results

results_ols = ols_from_scratch(X_train, y_train, X_test, y_test)

print("\n====== RESULTADOS DO OLS (DO ZERO) ======")
print("Coeficientes (beta):")
print(results_ols["beta"].flatten()) 
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

#%% 10-FOLD CROSS-VALIDATION — OLS DO ZERO

def kfold_indices(n_samples, k=10, seed=42):
    """Gera índices embaralhados divididos em K folds."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return np.array_split(indices, k)

folds = kfold_indices(len(X_train), k=10)

rmse_folds = []
r2_folds = []

for i, test_idx in enumerate(folds):
    print(f"Rodando fold {i+1}/10...")

    train_idx = np.setdiff1d(np.arange(len(X_train)), test_idx)

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]

    X_val = X_train.iloc[test_idx]
    y_val = y_train.iloc[test_idx]

    results_fold = ols_from_scratch(X_tr, y_tr, X_val, y_val)

    rmse_folds.append(results_fold["test_rmse"])
    r2_folds.append(results_fold["test_r2"])

# Resultados finais
rmse_cv_mean = np.mean(rmse_folds)
rmse_cv_std  = np.std(rmse_folds)

r2_cv_mean = np.mean(r2_folds)
r2_cv_std  = np.std(r2_folds)

print("\n====== 10-FOLD CV — OLS DO ZERO ======")
print(f"RMSE médio: {rmse_cv_mean:.6f}  (± {rmse_cv_std:.6f})")
print(f"R² médio  : {r2_cv_mean:.6f}  (± {r2_cv_std:.6f})")
print("========================================")

#%% 10-FOLD CROSS-VALIDATION — OLS SKLEARN

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

kf = KFold(n_splits=10, shuffle=True, random_state=42)

ols_model = LinearRegression()


r2_scores = cross_val_score(ols_model, X_train, y_train, 
                            scoring='r2', cv=kf)

rmse_scores = -cross_val_score(ols_model, X_train, y_train,
                               scoring='neg_root_mean_squared_error', 
                               cv=kf)

print("\n====== 10-FOLD CV — OLS (SKLEARN) ======")
print(f"RMSE médio: {rmse_scores.mean():.6f}  (± {rmse_scores.std():.6f})")
print(f"R² médio  : {r2_scores.mean():.6f}  (± {r2_scores.std():.6f})")
print("========================================")
#%% COMPARAÇÃO FINAL DO CROSS-VALIDATION

print("\n=========== COMPARAÇÃO CV ===========")
print(f"RMSE CV (do zero):   {rmse_cv_mean:.6f}  (± {rmse_cv_std:.6f})")
print(f"RMSE CV (sklearn):   {rmse_scores.mean():.6f}  (± {rmse_scores.std():.6f})")

print(f"R² CV (do zero):     {r2_cv_mean:.6f}  (± {r2_cv_std:.6f})")
print(f"R² CV (sklearn):     {r2_scores.mean():.6f}  (± {r2_scores.std():.6f})")
print("======================================")

