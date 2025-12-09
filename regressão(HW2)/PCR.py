#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

print("Imports OK")

# ATENÇÃO!!!! se der erro nessa parte do código, provavelmente sera por conta do diretorio em que se encontra os arquivos de teste e de treino na sua maquina, caso isso aconteça pedimos encarecidamente que cole o diretório em que os arquivos "treino.csv" e "teste.csv" se encontram na sua maquina no trecho de codigo abaixo, agradeçemos desde já.

df_train = pd.read_csv(
    r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv"
)
df_test = pd.read_csv(
    r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv"
)

cols_to_drop = ["class", "plate", "MJD"]

X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
y_train = df_train["redshift"]

X_test  = df_test.drop(columns=cols_to_drop + ["redshift"])
y_test  = df_test["redshift"]

print("Dados carregados e pré-processados.")

#  PADRONIZAÇÃO (OBRIGATÓRIA PARA PCA)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

ytr = np.array(y_train).ravel()
ytst = np.array(y_test).ravel()

print("Padronização concluída.")


#%%     PCR COMPLETA E CV


max_components = min(50, X_train_scaled.shape[1])  # 
n_splits = 10
random_state = 42

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

rmse_means = []
r2_means = []
components_list = list(range(1, max_components + 1))

print("Iniciando Cross-Validation para escolher número ótimo de componentes...")

for n_comp in components_list:

    pca = PCA(n_components=n_comp, random_state=random_state)
    Xtr_p = pca.fit_transform(X_train_scaled)

    model = LinearRegression()

    r2_scores = cross_val_score(model, Xtr_p, ytr, scoring='r2', cv=kf)

    mse_scores = -cross_val_score(model, Xtr_p, ytr,
                                  scoring='neg_mean_squared_error', cv=kf)
    rmse_scores = np.sqrt(mse_scores)

    rmse_means.append(rmse_scores.mean())
    r2_means.append(r2_scores.mean())

best_idx = int(np.argmin(rmse_means))
best_n = components_list[best_idx]

print(f"Melhor número de componentes (por RMSE CV): {best_n}")
#%% MOSTRAR MÉTRICAS MÉDIAS E DESVIOS DA CROSS-VALIDATION (PCR)

rmse_means_arr = np.array(rmse_means)
r2_means_arr   = np.array(r2_means)

rmse_mean_cv = rmse_means_arr.mean()
rmse_std_cv  = rmse_means_arr.std()

r2_mean_cv = r2_means_arr.mean()
r2_std_cv  = r2_means_arr.std()

print("=========== CROSS-VALIDATION PCR (10-fold) ===========")
print(f"Melhor número de componentes: {best_n}")
print("\n-- RMSE --")
print(f"RMSE médio (CV):      {rmse_mean_cv:.6f}")
print(f"Desvio do RMSE (CV):  {rmse_std_cv:.6f}")

print("\n-- R² --")
print(f"R² médio (CV):        {r2_mean_cv:.6f}")
print(f"Desvio do R² (CV):    {r2_std_cv:.6f}")
print("======================================================")


#%% PLOTS

plt.figure(figsize=(8,4))
plt.plot(components_list, rmse_means, marker='o')
plt.xlabel("Número de componentes")
plt.ylabel("RMSE médio (10-fold CV)")
plt.title("PCR — RMSE médio vs Número de Componentes")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(components_list, r2_means, marker='o')
plt.xlabel("Número de componentes")
plt.ylabel("R² médio (10-fold CV)")
plt.title("PCR — R² médio vs Número de Componentes")
plt.grid(True)
plt.show()


#%%    TREINO FINAL

pca_final = PCA(n_components=best_n, random_state=random_state)
Xtr_p_final = pca_final.fit_transform(X_train_scaled)
Xts_p_final = pca_final.transform(X_test_scaled)

model_final = LinearRegression()
model_final.fit(Xtr_p_final, ytr)

y_pred_train = model_final.predict(Xtr_p_final)
y_pred_test  = model_final.predict(Xts_p_final)

rmse_train = np.sqrt(mean_squared_error(ytr, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(ytst, y_pred_test))
r2_train = r2_score(ytr, y_pred_train)
r2_test  = r2_score(ytst, y_pred_test)

print("\n============== RESULTADOS PCR ==============")
print(f"n_components escolhido: {best_n}")
print(f"RMSE Treino: {rmse_train:.6f}")
print(f"RMSE Teste : {rmse_test:.6f}")
print(f"R² Treino  : {r2_train:.6f}")
print(f"R² Teste   : {r2_test:.6f}")
print("=============================================")


pca_explain = PCA().fit(X_train_scaled)
cumvar = np.cumsum(pca_explain.explained_variance_ratio_)

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
plt.xlabel("Número de componentes")
plt.ylabel("Variância explicada cumulativa")
plt.title("PCA — Variância Explicada Cumulativa")
plt.grid(True)
plt.show()

# %%import matplotlib.pyplot as plt


modelos = ["OLS", "Ridge", "PCR", "MLP"]

rmse_vals = [
    0.619526,   # OLS média
    0.6194,     # Ridge média
    0.638908,   # PCR
    0.5327      # MLP
]

r2_vals = [
    0.280607,   # OLS média
    0.2807,     # Ridge média
    0.234210,   # PCR
    0.4680      # MLP
]



plt.figure(figsize=(8, 4))
plt.bar(modelos, rmse_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
plt.ylabel("RMSE")
plt.title("Comparação de RMSE médio das Cross-Validations")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


plt.figure(figsize=(8, 4))
plt.bar(modelos, r2_vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
plt.ylabel("R²")
plt.title("Comparação de R² médio das Cross-Validations")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# %%
