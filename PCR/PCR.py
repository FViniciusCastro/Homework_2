#%% PCR DO ZERO + 10-FOLD CV (testar vários n_components)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os


train_path = r"C:\Users\carlo\Documentos\GitHub\Homework_2\data\treino.csv"
test_path  = r"C:\Users\carlo\Documentos\GitHub\Homework_2\data\teste.csv"

if 'X_train' not in globals() or 'y_train' not in globals():
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Arquivos de treino/teste não encontrados. Ajuste os paths.")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    cols_to_drop = ["class", "plate", "MJD"]
    X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
    y_train = df_train["redshift"]
    X_test  = df_test.drop(columns=cols_to_drop + ["redshift"])
    y_test  = df_test["redshift"]
    print("Dados carregados.")


scaler = StandardScaler()
Xtr = scaler.fit_transform(np.array(X_train))
Xtst = scaler.transform(np.array(X_test))
ytr = np.array(y_train).reshape(-1,1)
ytst = np.array(y_test).reshape(-1,1)

def compute_pca_components(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt
    scores = X @ Vt.T
    return components, scores, S

def ols_closed(X, y):
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return beta

def predict_ols(X, beta):
    return X @ beta

def kfold_indices(n, k=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


max_components = min(Xtr.shape[1], 30) 
n_components_list = list(range(1, max_components+1))
k = 10
folds = kfold_indices(len(Xtr), k=k)

rmse_mean_list = []
r2_mean_list = []

Xtr_centered = Xtr - Xtr.mean(axis=0)

components, scores_all, S = compute_pca_components(Xtr_centered)

for n_comp in n_components_list:
    rmse_folds = []
    r2_folds = []
    for fold in folds:
        val_idx = fold
        train_idx = np.setdiff1d(np.arange(len(Xtr)), val_idx)

        Xtrain_fold = Xtr[train_idx]
        Xval_fold   = Xtr[val_idx]

        mean_fold = Xtrain_fold.mean(axis=0)
        Xtrain_c = Xtrain_fold - mean_fold
        Xval_c   = Xval_fold - mean_fold

        _, scores_train, V = compute_pca_components(Xtrain_c)

        V = np.atleast_2d(V)

        max_comp_fold = min(n_comp, V.shape[0])

        P = V[:max_comp_fold, :].T 

        Z_train = Xtrain_c @ P
        Z_val   = Xval_c   @ P

        Z_train = Xtrain_c @ P
        Z_val   = Xval_c @ P

        Z_train_i = np.hstack((np.ones((Z_train.shape[0],1)), Z_train))
        Z_val_i   = np.hstack((np.ones((Z_val.shape[0],1)), Z_val))

        beta = ols_closed(Z_train_i, ytr[train_idx])
        yval_pred = predict_ols(Z_val_i, beta)

        rmse_folds.append(np.sqrt(mean_squared_error(ytr[val_idx], yval_pred)))
        r2_folds.append(r2_score(ytr[val_idx], yval_pred))

    rmse_mean_list.append(np.mean(rmse_folds))
    r2_mean_list.append(np.mean(r2_folds))

cv_results_pcr = {
    "n_components": n_components_list,
    "rmse_mean": rmse_mean_list,
    "r2_mean": r2_mean_list,
    "best_n": n_components_list[int(np.argmin(rmse_mean_list))]
}

print("PCR (do zero) - melhor número de componentes (por RMSE CV):", cv_results_pcr["best_n"])



#%% PCR COM SKLEARN (Pipeline) + CV

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

Xtr_arr = np.array(Xtr)
ytr_arr = np.array(ytr).ravel()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_means_sk = []
r2_means_sk = []

for n_comp in cv_results_pcr["n_components"]:
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(Xtr_arr)  
    
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(PCA(n_components=n_comp), LinearRegression())
    
    r2_scores = cross_val_score(pipe, Xtr_arr, ytr_arr, cv=kf, scoring="r2")
    
    try:
        rmse_scores = -cross_val_score(pipe, Xtr_arr, ytr_arr, cv=kf, scoring="neg_root_mean_squared_error")
    except Exception:
        mse_scores = -cross_val_score(pipe, Xtr_arr, ytr_arr, cv=kf, scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(mse_scores)
    r2_means_sk.append(r2_scores.mean())
    rmse_means_sk.append(rmse_scores.mean())

cv_results_pcr["rmse_mean_sk"] = rmse_means_sk
cv_results_pcr["r2_mean_sk"] = r2_means_sk
cv_results_pcr["best_n_sk"] = cv_results_pcr["n_components"][int(np.argmin(rmse_means_sk))]

print("PCR (sklearn) - melhor n_components (por RMSE CV):", cv_results_pcr["best_n_sk"])



#%% PLOTS RMSE and R2 vs n_components

import matplotlib.pyplot as plt

n = cv_results_pcr["n_components"]
rmse = cv_results_pcr["rmse_mean"]
r2 = cv_results_pcr["r2_mean"]

rmse_sk = cv_results_pcr.get("rmse_mean_sk", None)
r2_sk   = cv_results_pcr.get("r2_mean_sk", None)

plt.figure(figsize=(8,4))
plt.plot(n, rmse, marker='o', label='PCR do zero (RMSE)')
if rmse_sk is not None:
    plt.plot(n, rmse_sk, marker='x', label='PCR sklearn (RMSE)')
plt.xlabel("Número de componentes")
plt.ylabel("RMSE médio (CV)")
plt.title("PCR — RMSE médio (10-fold) vs número de componentes")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(n, r2, marker='o', label='PCR do zero (R2)')
if r2_sk is not None:
    plt.plot(n, r2_sk, marker='x', label='PCR sklearn (R2)')
plt.xlabel("Número de componentes")
plt.ylabel("R² médio (CV)")
plt.title("PCR — R² médio (10-fold) vs número de componentes")
plt.grid(True)
plt.legend()
plt.show()


#%% TREINO FINAL E AVALIAÇÃO NO TESTE

best_n = cv_results_pcr["best_n"]  
print("Usando n_components escolhido (do zero):", best_n)


from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

pca_final = PCA(n_components=best_n)
Z_train = pca_final.fit_transform(Xtr)
Z_test  = pca_final.transform(Xtst)


lr = LinearRegression()
lr.fit(Z_train, ytr.ravel())

yhat_train = lr.predict(Z_train).reshape(-1,1)
yhat_test  = lr.predict(Z_test).reshape(-1,1)

rmse_train = np.sqrt(mean_squared_error(ytr, yhat_train))
rmse_test  = np.sqrt(mean_squared_error(ytst, yhat_test))
r2_train = r2_score(ytr, yhat_train)
r2_test  = r2_score(ytst, yhat_test)

print("\n--- RESULTADOS FINAIS PCR ---")
print(f"n_components: {best_n}")
print(f"RMSE Treino: {rmse_train:.6f}")
print(f"RMSE Teste : {rmse_test:.6f}")
print(f"R² Treino  : {r2_train:.6f}")
print(f"R² Teste   : {r2_test:.6f}")
print("-----------------------------")
