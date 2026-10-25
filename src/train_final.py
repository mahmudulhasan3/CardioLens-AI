import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CLASS_NAMES, IMG_SIZE, ECGDataset, get_eval_transform
from src.model import build_model
from src.train import CSV_PATH, MODEL_DIR, ROOT, train_one_epoch

EPOCHS = 4  # chosen from the CV runs (val loss was lowest around epoch 4)
LR = 1e-4
BATCH_SIZE = 16


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Use ALL folds: every unique image goes into training
    ds = ECGDataset(
        CSV_PATH, ROOT, folds=[0, 1, 2, 3, 4], transform=get_eval_transform()
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print("training images:", len(ds))

    counts = ds.df["label"].value_counts()
    weights = torch.tensor(
        [len(ds) / (len(CLASS_NAMES) * counts[n]) for n in CLASS_NAMES],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, loader, criterion, optimizer, device)
        print(f"epoch {epoch} | train_loss {loss:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "resnet18_final.pt")

    # Metadata saved next to the weights; the inference code will read it
    meta = {
        "architecture": "resnet18",
        "class_names": CLASS_NAMES,
        "img_size": list(IMG_SIZE),
        "epochs": EPOCHS,
        "train_images": len(ds),
        "cv_macro_f1_mean": 0.873,
        "cv_macro_f1_std": 0.049,
    }
    (MODEL_DIR / "resnet18_final.json").write_text(json.dumps(meta, indent=2))
    print("saved: models/resnet18_final.pt + resnet18_final.json")


if __name__ == "__main__":
    main()
