import json
import sys
from pathlib import Path

import torch
from PIL import Image

from src.dataset import get_eval_transform
from src.model import build_model

# Project root, so paths work no matter where the program is started from
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


class ECGPredictor:
    """Loads the trained model once, then predicts for any number of images."""

    def __init__(
        self,
        weights_path=MODEL_DIR / "resnet18_final.pt",
        meta_path=MODEL_DIR / "resnet18_final.json",
    ):
        self.meta = json.loads(Path(meta_path).read_text())
        self.class_names = self.meta["class_names"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # pretrained=False: we load our own weights, no ImageNet download needed
        self.model = build_model(num_classes=len(self.class_names), pretrained=False)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()  # eval mode: no training behaviour
        self.transform = get_eval_transform()

    @torch.no_grad()  # no gradients needed for inference
    def predict(self, image):
        """image: a file path or a PIL.Image. Returns a plain dict."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)  # type: ignore
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().tolist()

        top = max(range(len(probs)), key=lambda i: probs[i])
        return {
            "predicted_class": self.class_names[top],
            "confidence": round(probs[top], 4),
            "probabilities": {
                name: round(p, 4) for name, p in zip(self.class_names, probs)
            },
        }


if __name__ == "__main__":
    # Usage: python -m src.predict "path/to/image.jpg"
    predictor = ECGPredictor()
    print(json.dumps(predictor.predict(sys.argv[1]), indent=2))
