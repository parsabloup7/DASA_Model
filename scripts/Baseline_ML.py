"""
Baseline Features & LightGBM Regression.
"""

from rdkit.Chem import AllChem
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import mean_absolute_error, roc_auc_score
import numpy as np

# Generate Morgan fingerprint from SMILES
def fp_from_smiles(sm):
    m = Chem.MolFromSmiles(sm)
    if m is None:
        return np.zeros(2048, dtype=np.int8)
    arr = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
    return np.array(arr)

# Compute fingerprints for all molecules
fps = np.stack(df['full_smiles'].map(fp_from_smiles).values)
X = fps  # baseline features (can extend with donor/acceptor fingerprints)
y_open = df['open_frac'].fillna(-1).values
mask_open = df['mask_open'].values
groups = df['core_smiles'].fillna(df['bridge_smiles']).astype(str).values  # grouping key for cross-validation

# Use only rows with valid open_frac labels
idx = np.where(mask_open)[0]
X_train = X[idx]
y_train = y_open[idx]
groups_train = groups[idx]

# Group K-Fold cross-validation setup
gkf = GroupKFold(n_splits=4)
scores = []

for tr, val in gkf.split(X_train, y_train, groups=groups_train):
    dtrain = lgb.Dataset(X_train[tr], label=y_train[tr])
    dval = lgb.Dataset(X_train[val], label=y_train[val])
    
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31
    }

    # Train LightGBM with early stopping
    bst = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=2000,
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = bst.predict(X_train[val])
    scores.append(mean_absolute_error(y_train[val], preds))

# Report cross-validated MAE
print("CV MAE open_frac:", np.mean(scores))
