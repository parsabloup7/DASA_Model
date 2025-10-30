""" 
Training loop (multi-task) for frozen Big_MPNN + new heads. 
"""
import time, math, os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Config ----
BATCH_SIZE = 8
EPOCHS = 80
LR = 3e-4
PATIENCE = 12
WEIGHT_DECAY = 1e-5
CHECKPOINT_PATH = "/kaggle/working/dasa_model_best2.pt"

# Filter dataframe: rows with at least one label
has_any_label = df['open_frac'].notna() | df['lambda_1_nm'].notna() | df['solvatochromic_slope_nm'].notna() | df['HighOpen'].notna() | df['switchability'].notna()
df_sub = df[has_any_label].reset_index(drop=True)
print("Using", len(df_sub), "rows with at least one label for training/validation.")

# Determine grouping key for GroupKFold
group_key = "core_smiles" if "core_smiles" in df_sub.columns else ("bridge_smiles" if "bridge_smiles" in df_sub.columns else None)
if group_key is None:
    df_sub["__grouptmp"] = np.arange(len(df_sub))
    group_key = "__grouptmp"

# Compute scalers for lambda and ss (used in collate_dasa_norm)
train_lambda_vals = df_sub['lambda_1_nm'].dropna().values
train_ss_vals = df_sub['solvatochromic_slope_nm'].dropna().values

if len(train_lambda_vals) > 0:
    LAMBDA_MEAN = float(np.mean(train_lambda_vals))
    LAMBDA_STD = float(np.std(train_lambda_vals)) if float(np.std(train_lambda_vals))>1e-6 else 1.0
else:
    LAMBDA_MEAN = 0.0; LAMBDA_STD = 1.0

if len(train_ss_vals) > 0:
    SS_MEAN = float(np.mean(train_ss_vals))
    SS_STD = float(np.std(train_ss_vals)) if float(np.std(train_ss_vals))>1e-6 else 1.0
else:
    SS_MEAN = 0.0; SS_STD = 1.0

print(f"Lambda scaler: mean={LAMBDA_MEAN:.3f}, std={LAMBDA_STD:.3f}; SS scaler: mean={SS_MEAN:.3f}, std={SS_STD:.3f}")

# ---- GroupKFold split (use first fold) ----
gkf = GroupKFold(n_splits=4)
groups = df_sub[group_key].astype(str).values
train_idx, val_idx = next(gkf.split(df_sub, df_sub.index, groups=groups))
train_df = df_sub.iloc[train_idx].reset_index(drop=True)
val_df = df_sub.iloc[val_idx].reset_index(drop=True)
print("Train / Val sizes:", len(train_df), len(val_df))

# ---- Datasets + DataLoaders ----
train_ds = DASADataset(train_df)
val_ds = DASADataset(val_df)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_dasa_norm, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_dasa_norm, num_workers=0)

# Move mpnn and model to device
mpnn.to(device); mpnn.eval()
model.to(device); model.train()

trainable_params = [p for p in model.parameters() if p.requires_grad]
print("Trainable params:", sum(p.numel() for p in trainable_params))
optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=4, verbose=True)

# ---- Helpers ----
def unnorm_lambda(tensor_norm):
    return tensor_norm * (LAMBDA_STD + 1e-12) + LAMBDA_MEAN

def unnorm_ss(tensor_norm):
    return tensor_norm * (SS_STD + 1e-12) + SS_MEAN

def batch_metrics_from_outputs(outputs, batch):
    """Return lists of raw-unit errors/labels from one batch (for reporting)."""
    res = {'open_errs': [], 'lambda_errs': [], 'ss_errs': [], 'high_preds': [], 'high_trues': [], 'switch_preds': [], 'switch_trues': []}
    targets_raw = batch['targets_raw']  # raw (not normalized), on device
    # open
    t_open = targets_raw['open_frac']; mask_open = ~torch.isnan(t_open)
    if mask_open.any() and 'open_mean' in outputs:
        m = outputs['open_mean'][mask_open].detach().cpu()
        t = t_open[mask_open].detach().cpu()
        res['open_errs'].extend(torch.abs(m - t).numpy().tolist())
    # lambda
    t_l = targets_raw['lambda_1']; mask_l = ~torch.isnan(t_l)
    if mask_l.any() and ('lambda_mean' in outputs or 'lambda' in outputs):
        key = 'lambda_mean' if 'lambda_mean' in outputs else 'lambda'
        pred_norm = outputs[key][mask_l].detach().cpu()
        pred_raw = unnorm_lambda(pred_norm)
        t = t_l[mask_l].detach().cpu()
        res['lambda_errs'].extend(torch.abs(pred_raw - t).numpy().tolist())
    # ss
    t_ss = targets_raw['ss']; mask_ss = ~torch.isnan(t_ss)
    if mask_ss.any() and 'ss_mean' in outputs:
        pred_norm = outputs['ss_mean'][mask_ss].detach().cpu()
        pred_raw = unnorm_ss(pred_norm)
        t = t_ss[mask_ss].detach().cpu()
        res['ss_errs'].extend(torch.abs(pred_raw - t).numpy().tolist())
    # highopen
    t_h = targets_raw['HighOpen']; mask_h = ~torch.isnan(t_h)
    if mask_h.any() and 'high_logit' in outputs:
        probs = torch.sigmoid(outputs['high_logit'][mask_h]).detach().cpu().numpy()
        res['high_preds'].extend((probs > 0.5).astype(int).tolist())
        res['high_trues'].extend(t_h[mask_h].detach().cpu().numpy().astype(int).tolist())
    # switch
    t_s = targets_raw['switchability']; mask_s = ~torch.isnan(t_s)
    if mask_s.any() and 'switch_logit' in outputs:
        probs = torch.sigmoid(outputs['switch_logit'][mask_s]).detach().cpu().numpy()
        res['switch_preds'].extend((probs > 0.5).astype(int).tolist())
        res['switch_trues'].extend(t_s[mask_s].detach().cpu().numpy().astype(int).tolist())
    return res

# ---- Training loop ----
best_val_loss = float("inf")
patience_ctr = 0
best_epoch = -1

for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    running_examples = 0
    accum_metrics = {'open_errs':[], 'lambda_errs':[], 'ss_errs':[], 'high_preds':[], 'high_trues':[], 'switch_preds':[], 'switch_trues':[]}

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)   # model expects batch dict with smiles lists + etn tensor
        loss_total, loss_dict = compute_multitask_loss(out, batch['targets'])
        if not loss_total.requires_grad:
            raise RuntimeError("loss_total does not require grad — check head params require_grad=True.")
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
        optimizer.step()

        running_loss += float(loss_total.item()) * len(batch['full_smiles'])
        running_examples += len(batch['full_smiles'])
        batch_met = batch_metrics_from_outputs(out, batch)
        for k,v in batch_met.items():
            accum_metrics[k].extend(v)

    train_loss_epoch = running_loss / (running_examples + 1e-12)

    # Validation
    model.eval()
    val_loss_accum = 0.0
    val_examples = 0
    val_metrics_acc = {'open_errs':[], 'lambda_errs':[], 'ss_errs':[], 'high_preds':[], 'high_trues':[], 'switch_preds':[], 'switch_trues':[]}
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss_total, loss_dict = compute_multitask_loss(out, batch['targets'])
            val_loss_accum += float(loss_total.item()) * len(batch['full_smiles'])
            val_examples += len(batch['full_smiles'])
            batch_met = batch_metrics_from_outputs(out, batch)
            for k,v in batch_met.items():
                val_metrics_acc[k].extend(v)

    val_loss_epoch = val_loss_accum / (val_examples + 1e-12)
    scheduler.step(val_loss_epoch)

    def mean_or_none(lst): return float(np.mean(lst)) if len(lst)>0 else None
    summary = {
        'train_loss': train_loss_epoch,
        'val_loss': val_loss_epoch,
        'open_mae': mean_or_none(val_metrics_acc['open_errs']),
        'lambda_mae': mean_or_none(val_metrics_acc['lambda_errs']),
        'ss_mae': mean_or_none(val_metrics_acc['ss_errs']),
        'high_acc': (np.mean(np.array(val_metrics_acc['high_preds']) == np.array(val_metrics_acc['high_trues'])) if len(val_metrics_acc['high_trues'])>0 else None),
        'switch_acc': (np.mean(np.array(val_metrics_acc['switch_preds']) == np.array(val_metrics_acc['switch_trues'])) if len(val_metrics_acc['switch_trues'])>0 else None)
    }

    print(f"Epoch {epoch:03d} | train_loss {summary['train_loss']:.4f} | val_loss {summary['val_loss']:.4f} | open_MAE {summary['open_mae']} | lambda_MAE {summary['lambda_mae']} | ss_MAE {summary['ss_mae']} | high_acc {summary['high_acc']} | switch_acc {summary['switch_acc']} | time {(time.time()-t0):.1f}s")

    # checkpoint
    if val_loss_epoch < best_val_loss - 1e-8:
        best_val_loss = val_loss_epoch
        best_epoch = epoch
        patience_ctr = 0
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "mpnn_state": {k:v for k,v in mpnn.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss_epoch,
            "L": L,
            "LAMBDA_MEAN": LAMBDA_MEAN, "LAMBDA_STD": LAMBDA_STD,
            "SS_MEAN": SS_MEAN, "SS_STD": SS_STD
        }, CHECKPOINT_PATH)
        print("  -> Saved new best model to", CHECKPOINT_PATH)
    else:
        patience_ctr += 1
        print(f"  no improvement (patience {patience_ctr}/{PATIENCE})")

    if patience_ctr >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training finished. Best epoch:", best_epoch, "best_val_loss:", best_val_loss)
print("Best model saved to", CHECKPOINT_PATH)
