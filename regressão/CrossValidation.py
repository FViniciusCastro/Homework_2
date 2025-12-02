# ATENÇÃO: antes de executar a cross validation deve-se executar as celulas do OLS.py
#%% 10-FOLD CROSS-VALIDATION — OLS DO ZERO

import numpy as np

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

# %%
