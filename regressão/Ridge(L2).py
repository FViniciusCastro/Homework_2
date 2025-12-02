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
#%%
# ============================================================
#                    RIDGE REGRESSION (DO ZERO)
# ============================================================

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

    # Matriz identidade para penalização
    I = np.eye(n_features)
    I[0, 0] = 0     # não penaliza o intercepto

    # Solução fechada da regressão Ridge
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

    # Convertendo tudo para arrays numpy
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_test  = np.asarray(y_test).reshape(-1, 1)

    # Adicionando intercepto
    X_train_i = add_intercept(X_train)
    X_test_i  = add_intercept(X_test)

    # Estimando coeficientes
    beta = ridge_beta(X_train_i, y_train, lam)

    # Previsões
    y_pred_train = predict(X_train_i, beta)
    y_pred_test  = predict(X_test_i, beta)

    # Resultados organizados
    return {
        "lambda": lam,
        "beta": beta,
        "train_rmse": rmse(y_train, y_pred_train),
        "test_rmse": rmse(y_test, y_pred_test),
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test)
    }


#%%

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


# %%
