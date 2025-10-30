"""
Data Normalization and Prep.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles
import numpy as np

# Load structures and measurements
structures = pd.read_csv('/kaggle/input/dasa-dataset/Structures.csv')  # adapt path as needed
measurements = pd.read_csv('/kaggle/input/dasa-dataset/Measurements.csv')

# Canonicalize SMILES helper function
def canon(sm):
    try:
        m = Chem.MolFromSmiles(sm)
        return MolToSmiles(m, canonical=True) if m is not None else sm
    except:
        return sm

# Apply canonicalization to relevant columns
for col in ['full_smiles','donor_smiles','acceptor_smiles','bridge_smiles','C2']:
    if col in structures.columns:
        structures[col] = structures[col].astype(str).map(canon)

# Merge measurements with structures
df = measurements.merge(structures, on='compound_id', how='left', suffixes=('_meas', '_struct'))
print("Columns after merge:", df.columns.tolist())

# Select the correct SMILES column
if 'full_smiles_struct' in df.columns:
    df['full_smiles'] = df['full_smiles_struct']
elif 'full_smiles_meas' in df.columns:
    df['full_smiles'] = df['full_smiles_meas']
else:
    raise KeyError("No full_smiles column found in merged DataFrame!")

# Map solvent names to ETN values
ETN_map = {'MeOH':0.762, 'DCM':0.309, 'CHCl3':0.259, 'ACN':0.460,
           'Tol':0.099, 'PhCl':0.188 , 'MeTHF':0.180}
df['solvent'] = df['open_form_solvent'].astype(str).str.strip()
df['ETN'] = df['solvent'].map(ETN_map).astype(float)

# Generate unique sample IDs
df['sample_id'] = df['compound_id'].astype(str) + '__' + df['solvent'].astype(str)

# Create masks for available measurements
df['mask_open'] = ~df['open_form_percentage'].isna()
df['mask_switch'] = ~df['switchability'].isna()
df['mask_lambda'] = ~df['lambda_1_nm'].isna()
df['mask_ss'] = ~df['solvatochromic_slope_nm'].isna()

# Normalize open-form percentage and classify high-open samples
df['open_frac'] = np.where(df['mask_open'],
                           df['open_form_percentage'].astype(float)/100.0,
                           np.nan)

THRESH_OPEN = 0.8  # Threshold for high-open classification
df['HighOpen'] = np.where(df['open_frac'] >= THRESH_OPEN, 1, 0)

# Display key columns (only those that exist)
cols_to_show = [
    'sample_id','full_smiles','donor_smiles','acceptor_smiles','bridge_smiles',
    'C2','solvent','ETN','open_frac','HighOpen','switchability',
    'lambda_1_nm','solvatochromic_slope_nm'
]
existing_cols = [c for c in cols_to_show if c in df.columns]
display(df[existing_cols].head())
