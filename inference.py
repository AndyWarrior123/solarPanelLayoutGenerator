import cv2
import numpy as np
import pandas as pd
import torch
from src.model   import SolarUNet
from src.utils   import load_config, load_checkpoint
from src.dataset import encode_meta
import os

cfg = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SolarUNet(cfg).to(device)
load_checkpoint("checkpoint/best.pt", model)
model.eval()

# img_path = os.path.join(self.image_dir, row["image_filename"])
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"Cannot open image: {img_path}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def predict(roof_path: str, meta_dict: dict) -> np.ndarray:
    img = cv2.imread(roof_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {roof_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    img_t = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
    meta = encode_meta(pd.Series(meta_dict)).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_t, meta)
    
    mask = torch.sigmoid(logit).squeeze().cpu().numpy() > cfg.inference.threshold
    return cv2.resize(mask.astype(np.uint8), (w, h))

if __name__ == "__main__":
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    mask = predict(
        roof_path="data/roofs/house_001_roof.jpg",
        meta_dict= {
            "roof_type": "tile", "connection_type": "single_phase",
            "num_panels": 13, "angle": 22, "num_strings": 1,
        }
    )
    cv2.imwrite(f"{cfg.inference.output_dir}/house_001_pred.png", mask * 255)
    print("Saved prediction.")
