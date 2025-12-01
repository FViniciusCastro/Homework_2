import numpy as np
import pandas as pd

# ============================================================
#               IMPLEMENTAÇÃO OLS DO ZERO
# ============================================================

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def compute_beta_ols(X, y):
    XT = X.T
    beta = np.linalg.inv(XT @ X) @ XT @ y
    return beta

def predict(X, beta):
    return X @ beta

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def ols_from_scratch(X_train, y_train, X_test, y_test):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    X_train_i = add_intercept(X_train)
    X_test_i = add_intercept(X_test)

    beta = compute_beta_ols(X_train_i, y_train)

    y_pred_train = predict(X_train_i, beta)
    y_pred_test = predict(X_test_i, beta)

    results = {
        "beta": beta,
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test)
    }

    return results

# ============================================================
#                  CARREGAR DADOS COM NOVOS NOMES
# ============================================================

df_train = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv")
df_test  = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv")

X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

