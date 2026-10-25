import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, ECGDataset, get_eval_transform
from src.model import build_model

# Run this file from the project root, so these paths are relative to it
ROOT = Path(".")
CSV_PATH = ROOT / "data" / "folds.csv"
MODEL_DIR = ROOT / "models"


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data. Returns the average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()  # no gradients needed during evaluation
def evaluate(model, loader, criterion, device):
    """Returns val loss, macro-F1, per-class recall, true labels, predictions."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    recalls = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    return avg_loss, macro_f1, recalls, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fold", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(42)  # reproducible runs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Train on the 4 other folds, validate on the chosen one
    train_folds = [f for f in range(5) if f != args.val_fold]
    train_ds = ECGDataset(CSV_PATH, ROOT, train_folds, get_eval_transform())
    val_ds = ECGDataset(CSV_PATH, ROOT, [args.val_fold], get_eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

    # Class weights computed from the TRAIN folds only
    counts = train_ds.df["label"].value_counts()
    weights = torch.tensor(
        [len(train_ds) / (len(CLASS_NAMES) * counts[n]) for n in CLASS_NAMES],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    MODEL_DIR.mkdir(exist_ok=True)
    save_path = MODEL_DIR / f"resnet18_fold{args.val_fold}.pt"
    best_f1, best_labels, best_preds = -1.0, None, None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, recalls, labels, preds = evaluate(
            model, val_loader, criterion, device
        )

        recall_text = " ".join(f"{n[:4]}={r:.2f}" for n, r in zip(CLASS_NAMES, recalls)) # type: ignore
        print(
            f"epoch {epoch:2d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | macro-F1 {val_f1:.3f} | {recall_text}"
        )

        # Keep the weights of the best epoch (by macro-F1)
        if val_f1 > best_f1:
            best_f1, best_labels, best_preds = val_f1, labels, preds
            torch.save(model.state_dict(), save_path)

    # Honest result: metrics of the LAST epoch (no picking the best epoch on val)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    result_path = out_dir / "cv_results.csv"
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["val_fold", "epochs", "lr", "macro_f1"]
                + [f"recall_{n}" for n in CLASS_NAMES]
            )
        writer.writerow(
            [args.val_fold, args.epochs, args.lr, f"{val_f1:.4f}"]
            + [f"{r:.4f}" for r in recalls] # type: ignore
        )
    print(f"Last-epoch macro-F1: {val_f1:.3f} (logged to {result_path})")

    print(f"\nBest macro-F1: {best_f1:.3f} (saved to {save_path})")
    print("Class order:", CLASS_NAMES)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(
        confusion_matrix(best_labels  , best_preds, labels=list # type: ignore
        (range(len(CLASS_NAMES))))
    )


if __name__ == "__main__":
    main()
