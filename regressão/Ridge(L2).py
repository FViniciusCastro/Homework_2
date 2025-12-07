#%%

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Imports OK")

# ATENÇÃO!!!! se der erro nessa parte do código, provavelmente sera por conta do diretorio em que se encontra os arquivos de teste e de treino na sua maquina, caso isso aconteça pedimos encarecidamente que cole o diretório em que os arquivos "treino.csv" e "teste.csv" se encontram na sua maquina no trecho de codigo abaixo, agradeçemos desde já.

df_train = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv")
df_test = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv")

cols_to_drop = ["class", "plate", "MJD"]


X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
y_train = df_train["redshift"]

X_test = df_test.drop(columns=cols_to_drop + ["redshift"])
y_test = df_test["redshift"]

print("Dados carregados e pré-processados.")
#%% RIDGE REGRESSION (DO ZERO)

def add_intercept(X):
    """
    Adiciona uma coluna de 1s (intercepto) à matriz de preditores X.
    """
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))


def ridge_beta(X, y, lam):
    """
    Calcula os coeficientes beta da Regressão Ridge usando a solução fechada:

        beta = (X^T X + λI)^(-1) X^T y

    Observação:
        O intercepto (primeira coluna) NÃO deve ser penalizado.
    """
    n_features = X.shape[1]

    I = np.eye(n_features)
    I[0, 0] = 0     

    return np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)


def predict(X, beta):
    """
    Gera previsões do modelo:
    y_pred = X * beta
    """
    return X @ beta


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).
    Mede o erro médio de previsão.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """
    Coeficiente de determinação R².
    Mede a proporção da variabilidade explicada pelo modelo.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def ridge_from_scratch(X_train, y_train, X_test, y_test, lam):
    """
    Executa todo o pipeline da Regressão Ridge:
        - adiciona intercepto
        - calcula beta
        - gera previsões no treino e no teste
        - calcula métricas (RMSE e R²)

    Retorna um dicionário com as métricas e coeficientes.
    """

    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test  = np.asarray(y_test).reshape(-1, 1)

    X_train_i = add_intercept(X_train)
    X_test_i  = add_intercept(X_test)

    beta = ridge_beta(X_train_i, y_train, lam)

    y_pred_train = predict(X_train_i, beta)
    y_pred_test  = predict(X_test_i, beta)

    return {
        "lambda": lam,
        "beta": beta,
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test)
    }

def print_ridge_results(results):
    print("\n--- RIDGE REGRESSION (DO ZERO) ---")
    print(f"Lambda (λ): {results['lambda']}")

    print("\nTreino:")
    print(f"  RMSE: {results['train_rmse']:.4f}")
    print(f"  R²:   {results['train_r2']:.4f}")

    print("\nTeste:")
    print(f"  RMSE: {results['test_rmse']:.4f}")
    print(f"  R²:   {results['test_r2']:.4f}")

    print("\nPrimeiros beta:")
    for i, b in enumerate(results["beta"][:5]):  
        print(f"  beta[{i}] = {float(b):.4f}")

    print("----------------------------------\n")


result = ridge_from_scratch(X_train, y_train, X_test, y_test, lam=10)
print_ridge_results(result)


#%% 10-FOLD CROSS-VALIDATION PARA RIDGE
import numpy as np

def kfold_indices(n_samples, k=10):
    """
    Gera índices embaralhados divididos em k folds.
    """
    indices = np.random.permutation(n_samples)
    return np.array_split(indices, k)


def ridge_cv_from_scratch(X, y, lambda_values, k=10):
    """
    Executa cross-validation do zero para vários valores de λ.
    
    Para cada λ:
        - cria K folds
        - treina o Ridge nos folds
        - calcula RMSE e R² médios
    """
    
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    folds = kfold_indices(len(X), k)

    rmse_means = []
    r2_means = []

    for lam in lambda_values:
        rmse_list = []
        r2_list = []

        for fold in folds:

            X_valid = X[fold]
            y_valid = y[fold]

            train_idx = np.setdiff1d(np.arange(len(X)), fold)
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]

            X_train_i = add_intercept(X_train_fold)
            X_valid_i = add_intercept(X_valid)

            beta = ridge_beta(X_train_i, y_train_fold, lam)

            y_pred = predict(X_valid_i, beta)

            rmse_list.append(rmse(y_valid, y_pred))
            r2_list.append(r2_score(y_valid, y_pred))

        rmse_means.append(np.mean(rmse_list))
        r2_means.append(np.mean(r2_list))

    results = {
        "lambda_values": lambda_values,
        "rmse_means": rmse_means,
        "r2_means": r2_means,
        "best_lambda": lambda_values[np.argmin(rmse_means)]
    }

    return results

def print_cv_results(cv_results):
    lambdas = cv_results["lambda_values"]
    rmse = cv_results["rmse_means"]
    r2 = cv_results["r2_means"]
    best_lam = cv_results["best_lambda"]

    print("\n--- RESULTADOS DO 10-FOLD CV (RIDGE DO ZERO) ---")
    print(f"{'λ':>8} | {'RMSE médio':>12} | {'R² médio':>10}")
    print("-" * 38)

    for lam, rm, r in zip(lambdas, rmse, r2):
        print(f"{lam:>8} | {rm:>12.4f} | {r:>10.4f}")

    print("\nMelhor λ encontrado:", best_lam)
    print("----------------------------------------------\n")

lambda_values = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 80, 100]

cv_results = ridge_cv_from_scratch(X_train, y_train, lambda_values, k=10)
print_cv_results(cv_results)

#%%
best_lam = cv_results["best_lambda"]
ridge_final = ridge_from_scratch(X_train, y_train, X_test, y_test, best_lam)
print_ridge_results(ridge_final)
#%% RIDGE COM SKLEARN (USANDO O MELHOR LAMBDA)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

best_lam = cv_results["best_lambda"]

ridge_sk = Ridge(alpha=best_lam)
ridge_sk.fit(X_train, y_train)

y_pred_train_sk = ridge_sk.predict(X_train)
y_pred_test_sk  = ridge_sk.predict(X_test)

rmse_train_sk = np.sqrt(mean_squared_error(y_train, y_pred_train_sk))
rmse_test_sk  = np.sqrt(mean_squared_error(y_test,  y_pred_test_sk))

r2_train_sk = r2_score(y_train, y_pred_train_sk)
r2_test_sk  = r2_score(y_test,  y_pred_test_sk)

print("\n--- RIDGE REGRESSION (SKLEARN) ---")
print(f"Lambda (λ): {best_lam}")

print("\nTreino:")
print(f"  RMSE: {rmse_train_sk:.4f}")
print(f"  R²:   {r2_train_sk:.4f}")

print("\nTeste:")
print(f"  RMSE: {rmse_test_sk:.4f}")
print(f"  R²:   {r2_test_sk:.4f}")

print("----------------------------------\n")

#%% CROSS-VALIDATION COM SKLEARN
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

best_lam = cv_results["best_lambda"]

ridge_sk = Ridge(alpha=best_lam)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_scores_sk = -cross_val_score(
    ridge_sk, X_train, y_train,
    scoring="neg_root_mean_squared_error",
    cv=kf
)

r2_scores_sk = cross_val_score(
    ridge_sk, X_train, y_train,
    scoring="r2",
    cv=kf
)

print("\n--- CROSS-VALIDATION COM SKLEARN (RIDGE) ---")

print("\nRMSE por fold:")
print(rmse_scores_sk)

print(f"\nRMSE médio: {rmse_scores_sk.mean():.4f}")
print(f"RMSE desvio: {rmse_scores_sk.std():.4f}")

print("\nR² por fold:")
print(r2_scores_sk)

print(f"\nR² médio: {r2_scores_sk.mean():.4f}")
print(f"R² desvio: {r2_scores_sk.std():.4f}")

print("----------------------------------\n")

# %% GRÁFICOS
import matplotlib.pyplot as plt

lambda_values = cv_results["lambda_values"]
rmse_means = cv_results["rmse_means"]
r2_means = cv_results["r2_means"]

# Gráfico RMSE vs lambda
plt.figure(figsize=(8,5))
plt.plot(lambda_values, rmse_means, marker='o')
plt.xscale('log')
plt.xlabel("Lambda (log scale)")
plt.ylabel("RMSE Médio (CV)")
plt.title("RMSE Médio vs Lambda (Ridge - Do Zero)")
plt.grid(True)
plt.show()

# Gráfico R² vs lambda
plt.figure(figsize=(8,5))
plt.plot(lambda_values, r2_means, marker='o')
plt.xscale('log')
plt.xlabel("Lambda (log scale)")
plt.ylabel("R² Médio (CV)")
plt.title("R² Médio vs Lambda (Ridge - Do Zero)")
plt.grid(True)
plt.show()

