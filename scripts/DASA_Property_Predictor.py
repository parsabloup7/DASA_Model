""" 
Simple DASA property prediction code. 
"""

import torch, numpy as np
from textwrap import dedent

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Helper functions -----
def to_float(x):
    """Convert tensor, ndarray, or numeric to float."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(x.item())
    return float(x)

def load_checkpoint_simple(checkpoint_path, mpnn, model):
    """Load model + MPNN checkpoint and return scalers."""
    ck = torch.load(checkpoint_path, map_location=device)
    mpnn.load_state_dict(ck.get("mpnn_state", {}), strict=False)
    model.load_state_dict(ck.get("model_state", {}), strict=False)
    scalers = {
        "LAMBDA_MEAN": ck.get("LAMBDA_MEAN", 587.52),
        "LAMBDA_STD":  ck.get("LAMBDA_STD", 36.05),
        "SS_MEAN":     ck.get("SS_MEAN", -20.48),
        "SS_STD":      ck.get("SS_STD", 16.44),
    }
    print("Model and MPNN restored from checkpoint.")
    return scalers

def unnorm(x, mean, std):
    """Un-normalize a tensor or float value."""
    return to_float(x) * (std + 1e-12) + mean

# ----- Simple prediction wrapper -----
def predict_dasa(smiles_full, etn_val=0.762):
    batch = {
        "full_smiles": [smiles_full],
        "donor_smiles": [""],
        "acceptor_smiles": [""],
        "bridge_smiles": [""],
        "C2_smiles": [""],
        "etn": torch.tensor([etn_val], dtype=torch.float32, device=device)
    }
    model.eval()
    with torch.no_grad():
        out = model(batch)

        lam_pred = unnorm(out["lambda_mean"], LAMBDA_MEAN, LAMBDA_STD)
        ss_pred  = unnorm(out["ss_mean"], SS_MEAN, SS_STD)
        open_frac = to_float(out["open_mean"])
        open_std  = float(torch.nn.functional.softplus(out["open_raw_sigma"]).cpu().item())
        high_prob = float(torch.sigmoid(out["high_logit"]).cpu().item())
        switch_prob = float(torch.sigmoid(out["switch_logit"]).cpu().item())

    print("\nPredicted DASA Properties")
    print("=" * 35)
    print(f"λ₁ (absorption max)    : {lam_pred:8.2f} nm")
    print(f"Slope (solvatochromic) : {ss_pred:8.2f} nm")
    print(f"Open fraction          : {open_frac:8.3f} ± {open_std:.3f}")
    print(f"High-open probability  : {high_prob:8.3f}")
    print(f"Switchability (binary) : {switch_prob:8.3f}")
    print("=" * 35)

# ---- Load model + checkpoint ----
checkpoint_path = "/kaggle/working/dasa_model_best2.pt"
scalers = load_checkpoint_simple(checkpoint_path, mpnn, model)

LAMBDA_MEAN, LAMBDA_STD = scalers["LAMBDA_MEAN"], scalers["LAMBDA_STD"]
SS_MEAN, SS_STD = scalers["SS_MEAN"], scalers["SS_STD"]

# ---- User input ----
smiles_input = input("\nEnter canonical DASA SMILES: ").strip()
predict_dasa(smiles_input)
