from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed class order. Never change it after training,
# otherwise saved models will predict the wrong names.
CLASS_NAMES = ["Abnormal", "History_of_MI", "MI", "Normal"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_SIZE = (256, 352)
# input size expected by common pretrained CNNs


def get_eval_transform():
    """Resize -> tensor -> normalize with ImageNet statistics."""
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ECGDataset(Dataset):
    """Loads the images of the chosen folds listed in folds.csv."""

    def __init__(self, csv_path, root_dir, folds, transform=None):
        df = pd.read_csv(csv_path)
        # Keep only rows that belong to the requested folds
        self.df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Paths in the CSV are relative to the project root
        image = Image.open(self.root_dir / row["path"]).convert("RGB")
        label = CLASS_TO_IDX[row["label"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
