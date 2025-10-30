"""
Featurizer / Adapter for BIG_MPNN
"""

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

# --- configuration ---
atom_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]  # selected atom types
bond_channel_names = ['Single', 'Double', 'Triple', 'Quadruple', 'Aromatic', 'Pi', 'Universal']
n_channels = len(bond_channel_names)  # 7
MAX_CAP = 128  # maximum atoms per molecule for padding

# Determine maximum atoms L from dataset (if df exists)
try:
    max_atoms_in_dataset = max([Chem.MolFromSmiles(s).GetNumAtoms() 
                                for s in df['full_smiles'].dropna().unique() 
                                if Chem.MolFromSmiles(s) is not None])
    L = min(max(max_atoms_in_dataset, 8), MAX_CAP)
    print("Detected max atoms in dataset:", max_atoms_in_dataset, " — using L =", L)
except Exception as e:
    L = 64
    print("Could not auto-compute max atoms; using default L =", L, "(error: {})".format(e))

# Map RDKit bond object to appropriate channels
def bond_to_channel(bond):
    """
    Returns list of channel indices for a bond.
    Considers bond type, aromaticity, and conjugation/pi bonds.
    """
    if bond is None:
        return []
    try:
        bt = bond.GetBondType()
        channels = []

        if bond.GetIsAromatic():
            channels.append(4)  # Aromatic channel
        else:
            if bt == Chem.rdchem.BondType.SINGLE:
                channels.append(0)
            elif bt == Chem.rdchem.BondType.DOUBLE:
                channels.append(1)
            elif bt == Chem.rdchem.BondType.TRIPLE:
                channels.append(2)
            elif bt == Chem.rdchem.BondType.QUADRUPLE:
                channels.append(3)
            else:
                channels.append(0)  # fallback for unknown bond type

        try:
            if bond.GetIsConjugated() and 5 not in channels:
                channels.append(5)  # Pi channel
        except Exception:
            pass  # ignore if GetIsConjugated unavailable

        return list(sorted(set(channels)))
    except Exception:
        return [0]  # safe fallback

# Featurize a single molecule (SMILES) -> (g, h)
def featurize_smiles(sm, L=L, atom_list=atom_list):
    m = Chem.MolFromSmiles(sm)
    if m is None:
        g = np.zeros((n_channels, L, L), dtype=np.float32)
        h = np.zeros((L, 3), dtype=np.float32)
        return g, h

    na = m.GetNumAtoms()
    if na > L:
        print(f"WARNING: molecule has {na} atoms > L={L}. Truncating to {L} atoms.")
        na = L

    g = np.zeros((n_channels, L, L), dtype=np.float32)
    h = np.zeros((L, 3), dtype=np.float32)  # Features: [atomic_number, is_aromatic, total_Hs]

    # Atom features
    for i, a in enumerate(m.GetAtoms()):
        if i >= L: break
        Z = a.GetAtomicNum()
        h[i, 0] = float(Z)                       # MUST: atomic number
        h[i, 1] = 1.0 if a.GetIsAromatic() else 0.0
        try:
            h[i, 2] = float(a.GetTotalNumHs())
        except Exception:
            h[i, 2] = float(a.GetNumImplicitHs()) if hasattr(a, "GetNumImplicitHs") else 0.0

    # Bond features
    for bond in m.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 >= L or a2 >= L:
            continue
        chs = bond_to_channel(bond) or []
        for ch in chs:
            g[ch, a1, a2] = 1.0
            g[ch, a2, a1] = 1.0

    # Universal channel
    any_bonds = (g.sum(axis=0) > 0).astype(np.float32)
    g[6] = any_bonds

    return g, h

# Featurize batch of SMILES
def featurize_batch(smiles_list, L=L, atom_list=atom_list):
    B = len(smiles_list)
    g_batch = np.zeros((B, n_channels, L, L), dtype=np.float32)
    h_batch = np.zeros((B, L, 3), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        g, h = featurize_smiles(s, L=L, atom_list=atom_list)
        g_batch[i] = g
        h_batch[i] = h
    return g_batch, h_batch

# Quick test on first few SMILES
test_smiles = df['full_smiles'].dropna().unique()[:4].tolist()
print("Testing featurizer on SMILES (first 4):", test_smiles)
g_b, h_b = featurize_batch(test_smiles, L=L)
print("g_batch shape:", g_b.shape, "h_batch shape:", h_b.shape)

# Convert to torch tensors for MPNN
import torch
g_torch = torch.tensor(g_b, dtype=torch.float32, device=device)
h_torch = torch.tensor(h_b, dtype=torch.float32, device=device)

print("Attempting to call mpnn with these tensors (testing shape compatibility)...")
try:
    with torch.no_grad():
        out = mpnn(g_torch, h_torch)
    print("MPNN forward returned. Output type:", type(out))
    if isinstance(out, torch.Tensor):
        print("Output shape:", out.shape)
    elif isinstance(out, (list, tuple)):
        print("Output lengths/shapes:", [x.shape if isinstance(x, torch.Tensor) else type(x) for x in out])
    else:
        print("Output (object):", out)
    print("SUCCESS: featurizer produced shapes MPNN accepted.")
except Exception as e:
    print("MPNN forward failed with error:", type(e).__name__, str(e))
    print("g_torch.shape:", g_torch.shape, "h_torch.shape:", h_torch.shape)
