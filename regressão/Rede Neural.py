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
#%% CURVA DE APRENDIZADO (LOSS)

plt.figure(figsize=(8,4))
plt.plot(mlp.loss_curve_)
plt.title("Curva de aprendizado")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# %%
