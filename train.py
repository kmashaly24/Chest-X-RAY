"""End-to-end training script.

Flow: (optionally) download the data -> split by patient -> train the ViT ->
save the best checkpoint and the per-class F1 thresholds. No visualization is
produced; only metrics are printed to the console.

Real training (GPU):
    python train.py --download-path data --output-dir checkpoints --epochs 15

Smoke test on any machine (tiny synthetic data, no download / no pretraining):
    python train.py --smoke --epochs 1 --batch-size 8 --no-pretrain
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import config
import data_utils
from model import AsymmetricLoss, ChestViT

# ---- hyperparameters (mirror the notebook) ----
EPOCHS = 15
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
GRAD_CLIP = 1.0
LOG_EVERY = 100


def compute_metrics(targets, preds):
    aps, aucs = [], []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(targets[:, c], preds[:, c]))
        try:
            aucs.append(roc_auc_score(targets[:, c], preds[:, c]))
        except ValueError:
            pass
    return float(np.mean(aps)) if aps else 0.0, float(np.mean(aucs)) if aucs else 0.0


def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_amp, clip=GRAD_CLIP, limit_batches=None):
    model.train()
    running, all_p, all_t = 0.0, [], []
    for i, (images, labels) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running += loss.item()
        all_p.append(torch.sigmoid(out).detach().float().cpu().numpy())
        all_t.append(labels.detach().float().cpu().numpy())
        if (i + 1) % LOG_EVERY == 0:
            print(f"  [train {i + 1}] loss {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

    n = len(loader) if limit_batches is None else min(len(loader), limit_batches)
    mAP, mAUC = compute_metrics(np.vstack(all_t), np.vstack(all_p))
    return running / max(1, n), mAP, mAUC


@torch.inference_mode()
def eval_one_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    running, all_p, all_t = 0.0, [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(images)
            loss = criterion(out, labels)
        running += loss.item()
        all_p.append(torch.sigmoid(out).float().cpu().numpy())
        all_t.append(labels.float().cpu().numpy())

    mAP, mAUC = compute_metrics(np.vstack(all_t), np.vstack(all_p))
    return running / len(loader), mAP, mAUC, np.vstack(all_p), np.vstack(all_t)


def compute_thresholds(targets, preds, num_classes):
    thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        y, p = targets[:, c], preds[:, c]
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 33):
            try:
                f = f1_score(y, (p >= t).astype(int), zero_division=0)
            except ValueError:
                f = 0.0
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[c] = best_t
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Train the Chest X-ray ViT classifier.")
    parser.add_argument("--download-path", default="data", help="Folder for data.zip and the CSV.")
    parser.add_argument("--output-dir", default="checkpoints", help="Where to save the model and thresholds.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (use 0 on Windows).")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--limit-batches", type=int, default=None, help="Limit train batches per epoch (debug).")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate on the held-out test set at the end.")
    parser.add_argument("--skip-download", action="store_true", help="Use data already present in --download-path.")
    parser.add_argument("--no-pretrain", action="store_true", help="Skip ImageNet pretrained weights (offline/smoke).")

    args = parser.parse_args()

    device = config.get_device()
    use_amp = device.type == "cuda"
    print(f"Using device: {device} (AMP: {use_amp})")

    # ---- data ----
    if not args.skip_download:
        print("Downloading data from Kaggle ...")
        data_utils.download_data(args.download_path)

    zip_path = os.path.join(args.download_path, "data.zip")
    csv_path = os.path.join(args.download_path, "Data_Entry_2017.csv")
    if not os.path.exists(csv_path):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"{zip_path} not found. Provide Kaggle credentials to download, "
                f"prepare the data manually, or use --smoke for a synthetic run.")
        csv_path = data_utils.extract_csv(zip_path, args.download_path)

    df = data_utils.build_dataframe(csv_path, zip_path)
    train_df, val_df, test_df = data_utils.patient_split(df)
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    train_loader, val_loader, test_loader = data_utils.build_loaders(
        train_df, val_df, test_df, args.download_path, args.batch_size, args.workers)

    # ---- model / optimizer / scheduler / scaler ----
    model = ChestViT(num_classes=config.NUM_CLASSES, pretrained=not args.no_pretrain).to(device)
    criterion = AsymmetricLoss()

    head_params = [p for n, p in model.named_parameters() if "heads.head" in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if "heads.head" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": BACKBONE_LR},
         {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"Total steps {total_steps} | warmup {warmup_steps}")

    # ---- output paths ----
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "run_best.pth")
    checkpoint_path = os.path.join(args.output_dir, "run_checkpoint.pth")
    thresholds_path = os.path.join(args.output_dir, "f1_thresholds.npy")

    best_mAP, patience_counter = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "train_mAP": [], "val_mAP": [], "val_mAUC": []}

    print("Training started...")
    for epoch in range(args.epochs):
        print("=" * 60)
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print("=" * 60)

        tr_loss, tr_mAP, _ = train_one_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, use_amp, limit_batches=args.limit_batches)
        va_loss, va_mAP, va_mAUC, _, _ = eval_one_epoch(model, val_loader, criterion, device, use_amp)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_mAP"].append(tr_mAP)
        history["val_mAP"].append(va_mAP)
        history["val_mAUC"].append(va_mAUC)
        print(f"Epoch {epoch + 1} | train_loss {tr_loss:.4f} mAP {tr_mAP:.4f} | "
              f"val_loss {va_loss:.4f} mAP {va_mAP:.4f} AUC {va_mAUC:.4f}")

        if va_mAP > best_mAP:
            best_mAP, patience_counter = va_mAP, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "mAP": va_mAP}, best_path)
            print(f"  New best val mAP {va_mAP:.4f} -> saved best model.")
        else:
            patience_counter += 1
            print(f"  No mAP improvement. patience {patience_counter}/{args.patience}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_mAP": best_mAP, "patience_counter": patience_counter,
                    "history": history}, checkpoint_path)

        if patience_counter >= args.patience:
            print("Early stopping (val mAP stagnated).")
            break

    # ---- thresholds from the validation set ----
    print("\nLoading best model and computing F1 thresholds on validation set ...")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    _, _, _, val_preds, val_targets = eval_one_epoch(model, val_loader, criterion, device, use_amp)
    thresholds = compute_thresholds(val_targets, val_preds, config.NUM_CLASSES)
    np.save(thresholds_path, thresholds)
    print(f"Saved F1 thresholds to {thresholds_path}")

    # ---- optional test evaluation ----
    if args.eval_test:
        _, te_mAP, te_mAUC, test_preds, test_targets = eval_one_epoch(model, test_loader, criterion, device, use_amp)
        print(f"TEST  mAP {te_mAP:.4f} | macro-AUC {te_mAUC:.4f}")
        y_bin = (test_preds >= thresholds).astype(int)
        print(f"Micro-F1 {f1_score(test_targets, y_bin, average='micro', zero_division=0):.4f} | "
              f"Macro-F1 {f1_score(test_targets, y_bin, average='macro', zero_division=0):.4f}")

    print(f"\nTraining finished. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()