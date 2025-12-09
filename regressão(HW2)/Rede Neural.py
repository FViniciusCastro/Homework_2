#%%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

print("Imports OK")

# ATENÇÃO!!!! se der erro nessa parte do código, provavelmente sera por conta do diretorio em que se encontra os arquivos de teste e de treino na sua maquina, caso isso aconteça pedimos encarecidamente que cole o diretório em que os arquivos "treino.csv" e "teste.csv" se encontram na sua maquina no trecho de codigo abaixo, agradeçemos desde já.

df_train = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\treino.csv")
df_test = pd.read_csv(r"C:\Users\vinic\OneDrive\Documentos\MeusProjetos\Homework_2\data\teste.csv")

cols_to_drop = ["class", "plate", "MJD"]

X_train = df_train.drop(columns=cols_to_drop + ["redshift"])
y_train = df_train["redshift"]

X_test = df_test.drop(columns=cols_to_drop + ["redshift"])
y_test = df_test["redshift"]

print("Dados carregados.")


#%% MODELO DE REDE NEURAL 
pt = PowerTransformer(method="yeo-johnson")
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pt)
X_test_scaled = scaler.transform(X_test_pt)


mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.0008,
    batch_size=64,
    max_iter=300,
    alpha=0.0001,            
    early_stopping=True,    
    n_iter_no_change=20,     
    random_state=42,
    verbose=True
)

print("Treinando a rede neural...")
mlp.fit(X_train_scaled, y_train)

print("Treinamento concluído.")

#%% RESULTADOS

y_pred_train = mlp.predict(X_train_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)

print("\n========== RESULTADOS REDE NEURAL ==========")
print("Treino:")
print(f"  RMSE: {rmse_train:.4f}")
print(f"  R²:   {r2_train:.4f}\n")

print("Teste:")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  R²:   {r2_test:.4f}")
print("============================================")
#%% CROSS-VALIDATION PARA A REDE NEURAL (10-fold)
# ATENÇÃO: essa celula demora MUITO para rodar e não é obrigatória
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

print("Iniciando 10-fold cross-validation para a MLP...")

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
    
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    pt_cv = PowerTransformer(method="yeo-johnson")
    X_tr_pt = pt_cv.fit_transform(X_tr)
    X_val_pt = pt_cv.transform(X_val)

    scaler_cv = StandardScaler()
    X_tr_scaled = scaler_cv.fit_transform(X_tr_pt)
    X_val_scaled = scaler_cv.transform(X_val_pt)

    mlp_cv = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0008,
        batch_size=64,
        max_iter=300,
        alpha=0.0001,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )

    mlp_cv.fit(X_tr_scaled, y_tr)

    y_val_pred = mlp_cv.predict(X_val_scaled)

    rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2_fold = r2_score(y_val, y_val_pred)

    rmse_scores.append(rmse_fold)
    r2_scores.append(r2_fold)

    print(f"Fold {fold}: RMSE={rmse_fold:.4f} | R²={r2_fold:.4f}")

# Resultados finais
rmse_mean = np.mean(rmse_scores)
rmse_std = np.std(rmse_scores)

r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)

print("\n========== RESULTADOS CROSS-VALIDATION (MLP) ==========")
print(f"RMSE médio (10-fold): {rmse_mean:.4f}  ± {rmse_std:.4f}")
print(f"R² médio (10-fold):   {r2_mean:.4f}  ± {r2_std:.4f}")
print("=======================================================\n")

#%% CURVA DE APRENDIZADO (LOSS)

plt.figure(figsize=(8,4))
plt.plot(mlp.loss_curve_)
plt.title("Curva de aprendizado")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()
