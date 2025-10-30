"""
Fusion + Multitask Heads, Dataset, Collate, Loss Helpers, Quick Inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

# --- Multi-head DASA model ---
class DASA_MultiHead(nn.Module):
    """
    Multi-task model with frozen Big_MPNN backbone producing per-atom embeddings (B, L, D).
    Embeddings are pooled to molecule-level and fused with full+D+A+Bridge+C2+ETN features.

    Heads:
      - Regression: mean, logvar, raw_sigma  (open, lambda, ss)
      - Classification: high_logit, switch_logit
    """
    def __init__(self, mpnn_backbone, message_size=128, freeze_backbone=True, dropout=0.2):
        super().__init__()
        self.mpnn = mpnn_backbone
        self.D = message_size
        if freeze_backbone:
            for p in self.mpnn.parameters():
                p.requires_grad = False

        fusion_dim = self.D * 5 + 1  # full + donor + acceptor + bridge + C2 + ETN
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        def make_mean_logvar_sigma():
            mean = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
            logvar = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
            raw_sigma = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
            return mean, logvar, raw_sigma

        # Regression heads
        self.open_mean, self.open_logvar, self.open_raw_sigma = make_mean_logvar_sigma()
        self.lambda_mean, self.lambda_logvar, self.lambda_raw_sigma = make_mean_logvar_sigma()
        self.ss_mean, self.ss_logvar, self.ss_raw_sigma = make_mean_logvar_sigma()

        # Classification heads
        self.high_logit = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.switch_logit = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def pool(self, atom_emb):
        # Sum pooling from atom-level to molecule-level (B, D)
        return atom_emb.sum(dim=1)

    def embed_smiles_list(self, smiles_list):
        """
        Featurizes SMILES list using featurize_batch() and returns atom-level embeddings (B, L, D)
        """
        if len(smiles_list) == 0:
            return torch.zeros((0, L, self.D), device=device, dtype=torch.float32)
        g_np, h_np = featurize_batch(smiles_list, L=L)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        h = torch.tensor(h_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            atom_emb = self.mpnn(g, h)
        return atom_emb

    def forward(self, batch):
        # Embed components
        full_emb_atoms = self.embed_smiles_list(batch['full_smiles'])
        d_emb_atoms = self.embed_smiles_list(batch['donor_smiles'])
        a_emb_atoms = self.embed_smiles_list(batch['acceptor_smiles'])
        b_emb_atoms = self.embed_smiles_list(batch['bridge_smiles'])
        c2_emb_atoms = self.embed_smiles_list(batch['C2_smiles'])

        # Pool and fuse features
        z_full = self.pool(full_emb_atoms)
        z_d = self.pool(d_emb_atoms)
        z_a = self.pool(a_emb_atoms)
        z_b = self.pool(b_emb_atoms)
        z_c2 = self.pool(c2_emb_atoms)
        etn = batch['etn'].view(-1, 1)

        z = torch.cat([z_full, z_d, z_a, z_b, z_c2, etn], dim=1)
        z = self.fusion(z)

        # Outputs
        outputs = {}
        outputs['open_mean'] = self.open_mean(z).squeeze(-1)
        outputs['open_logvar'] = self.open_logvar(z).squeeze(-1)
        outputs['open_raw_sigma'] = self.open_raw_sigma(z).squeeze(-1)
        outputs['open_std'] = outputs['open_raw_sigma']  # alias

        outputs['lambda_mean'] = self.lambda_mean(z).squeeze(-1)
        outputs['lambda_logvar'] = self.lambda_logvar(z).squeeze(-1)
        outputs['lambda_raw_sigma'] = self.lambda_raw_sigma(z).squeeze(-1)
        outputs['lambda'] = outputs['lambda_mean']  # alias

        outputs['ss_mean'] = self.ss_mean(z).squeeze(-1)
        outputs['ss_logvar'] = self.ss_logvar(z).squeeze(-1)
        outputs['ss_raw_sigma'] = self.ss_raw_sigma(z).squeeze(-1)
        outputs['ss'] = outputs['ss_mean']  # alias

        outputs['high_logit'] = self.high_logit(z).squeeze(-1)
        outputs['switch_logit'] = self.switch_logit(z).squeeze(-1)

        outputs['embedding'] = z
        return outputs

# --- Dataset class ---
class DASADataset(Dataset):
    """Simple dataset wrapper for DASA compounds"""
    def __init__(self, df_in):
        self.df = df_in.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return {
            'full_smiles': str(r.get('full_smiles','')),
            'donor_smiles': str(r.get('donor_smiles','')),
            'acceptor_smiles': str(r.get('acceptor_smiles','')),
            'bridge_smiles': str(r.get('bridge_smiles','')),
            'C2_smiles': str(r.get('C2','')),
            'etn': float(r.get('ETN', 0.0)) if not pd.isna(r.get('ETN', np.nan)) else 0.0,
            'open_frac': float(r.open_frac) if not pd.isna(r.open_frac) else np.nan,
            'switchability': float(r.switchability) if not pd.isna(r.switchability) else np.nan,
            'lambda_1_nm': float(r.lambda_1_nm) if not pd.isna(r.lambda_1_nm) else np.nan,
            'ss': float(r.solvatochromic_slope_nm) if not pd.isna(r.solvatochromic_slope_nm) else np.nan,
            'HighOpen': float(r.HighOpen) if not pd.isna(r.HighOpen) else np.nan
        }

# --- Collate function (with optional normalization) ---
def collate_dasa_norm(batch):
    """
    Prepares batch dictionary for training:
      - SMILES components
      - ETN tensor
      - normalized & raw targets for multitask learning
    Global scalers LAMBDA_MEAN/STD and SS_MEAN/STD are applied if present.
    """
    full_smiles = [b['full_smiles'] for b in batch]
    donor_smiles = [b['donor_smiles'] for b in batch]
    acceptor_smiles = [b['acceptor_smiles'] for b in batch]
    bridge_smiles = [b['bridge_smiles'] for b in batch]
    c2_smiles = [b['C2_smiles'] for b in batch]
    etn = torch.tensor([b['etn'] for b in batch], dtype=torch.float32, device=device)

    # Raw targets (NaNs retained)
    open_raw = torch.tensor([b['open_frac'] if not (b['open_frac'] is np.nan) else np.nan for b in batch], dtype=torch.float32)
    switch_raw = torch.tensor([b['switchability'] if not (b['switchability'] is np.nan) else np.nan for b in batch], dtype=torch.float32)
    lambda_raw = torch.tensor([b['lambda_1_nm'] if not (b['lambda_1_nm'] is np.nan) else np.nan for b in batch], dtype=torch.float32)
    ss_raw = torch.tensor([b['ss'] if not (b['ss'] is np.nan) else np.nan for b in batch], dtype=torch.float32)
    high_raw = torch.tensor([b['HighOpen'] if not (b['HighOpen'] is np.nan) else np.nan for b in batch], dtype=torch.float32)

    # Normalized copies (applies global scalers if defined)
    lambda_norm = lambda_raw.clone()
    ss_norm = ss_raw.clone()
    if 'LAMBDA_MEAN' in globals() and 'LAMBDA_STD' in globals():
        lambda_norm = (lambda_raw - LAMBDA_MEAN) / (LAMBDA_STD + 1e-12)
    if 'SS_MEAN' in globals() and 'SS_STD' in globals():
        ss_norm = (ss_raw - SS_MEAN) / (SS_STD + 1e-12)

    # Move tensors to device
    targets = {
        'open_frac': open_raw.to(device),
        'switchability': switch_raw.to(device),
        'lambda_1': lambda_norm.to(device),
        'ss': ss_norm.to(device),
        'HighOpen': high_raw.to(device)
    }
    targets_raw = {
        'open_frac': open_raw.to(device),
        'switchability': switch_raw.to(device),
        'lambda_1': lambda_raw.to(device),
        'ss': ss_raw.to(device),
        'HighOpen': high_raw.to(device)
    }

    batch_dict = {
        'full_smiles': full_smiles,
        'donor_smiles': donor_smiles,
        'acceptor_smiles': acceptor_smiles,
        'bridge_smiles': bridge_smiles,
        'C2_smiles': c2_smiles,
        'etn': etn,
        'targets': targets,
        'targets_raw': targets_raw
    }
    return batch_dict

# --- Loss helpers ---
EPS = 1e-6
PI = np.pi

def heteroscedastic_nll(pred_mean, pred_logvar, target, mask):
    """
    Heteroscedastic negative log-likelihood for regression targets
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_mean.device, requires_grad=True)
    sigma = F.softplus(pred_logvar) + EPS
    var = sigma * sigma
    nll = 0.5 * torch.log(2 * PI * var[mask]) + 0.5 * ((target[mask] - pred_mean[mask]) ** 2) / var[mask]
    return nll.mean()

bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

def compute_multitask_loss(outputs, targets):
    """
    Computes total multitask loss combining heteroscedastic NLL for regression
    and BCE for classification heads.
    """
    device_loc = outputs['open_mean'].device
    mask_open = ~torch.isnan(targets['open_frac'])
    mask_lambda = ~torch.isnan(targets['lambda_1'])
    mask_ss = ~torch.isnan(targets['ss'])
    mask_switch = ~torch.isnan(targets['switchability'])
    mask_high = ~torch.isnan(targets['HighOpen'])

    loss_open = heteroscedastic_nll(outputs['open_mean'], outputs['open_logvar'], targets['open_frac'], mask_open)
    loss_lambda = heteroscedastic_nll(outputs['lambda_mean'], outputs['lambda_logvar'], targets['lambda_1'], mask_lambda)
    loss_ss = heteroscedastic_nll(outputs['ss_mean'], outputs['ss_logvar'], targets['ss'], mask_ss)

    if mask_switch.sum() > 0:
        switch_losses = bce_loss_fn(outputs['switch_logit'], targets['switchability'])
        loss_switch = switch_losses[mask_switch].mean()
    else:
        loss_switch = torch.tensor(0.0, device=device_loc, requires_grad=True)

    if mask_high.sum() > 0:
        high_losses = bce_loss_fn(outputs['high_logit'], targets['HighOpen'])
        loss_high = high_losses[mask_high].mean()
    else:
        loss_high = torch.tensor(0.0, device=device_loc, requires_grad=True)

    total_loss = (1.0 * loss_open) + (0.5 * loss_lambda) + (0.5 * loss_ss) + (1.0 * loss_switch) + (0.5 * loss_high)

    loss_dict = {
        'total_loss': total_loss,
        'loss_open': loss_open.detach() if isinstance(loss_open, torch.Tensor) else loss_open,
        'loss_lambda': loss_lambda.detach() if isinstance(loss_lambda, torch.Tensor) else loss_lambda,
        'loss_ss': loss_ss.detach() if isinstance(loss_ss, torch.Tensor) else loss_ss,
        'loss_switch': loss_switch.detach() if isinstance(loss_switch, torch.Tensor) else loss_switch,
        'loss_high': loss_high.detach() if isinstance(loss_high, torch.Tensor) else loss_high
    }
    return total_loss, loss_dict

# --- Instantiate skeleton model ---
model = DASA_MultiHead(mpnn_backbone=mpnn, message_size=mpnn.message_size if hasattr(mpnn,'message_size') else 128).to(device)
print("Model instantiated. Params (trainable):", sum(p.numel() for p in model.parameters() if p.requires_grad))
