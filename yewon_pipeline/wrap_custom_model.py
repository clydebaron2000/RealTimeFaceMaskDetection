import os
import torch
from dataclasses import asdict

from pipeline_1 import TrainConfig, CustomCNN

# 1) Original checkpoint path
RAW_CKPT = "best_pytorch_model_custom.pth"

# 2) Output directory in runs folder
OUT_DIR = "runs_12k/cylde_custom"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "best.pth")

# 3) Load state_dict
state_dict = torch.load(RAW_CKPT, map_location="cpu")

# 4) Infer number of classes from fc layer weight shape
num_classes = state_dict["fc.weight"].shape[0]
print("num_classes detected from fc.weight:", num_classes)

# 5) Load state_dict into CustomCNN instance
model = CustomCNN(in_channels=3, num_classes=num_classes)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing keys:", missing)
print("unexpected keys:", unexpected)

# 6) Create TrainConfig
cfg = TrainConfig(
    data_root="/local-ssd/yl3427/test/input/fm_12k",
    output_dir=OUT_DIR,
    dataset_mode="mask12k",
    model_name="clyde_custom_cnn",
    num_classes=num_classes,
    merge_incorrect_with_nomask=True,
    pretrained=False,
    freeze_backbone=False,
    epochs=0,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    num_workers=4,
    img_size=40,
    seed=42,
    amp=True,
)

wrapped = {
    "epoch": 0,
    "model_state": model.state_dict(),
    "optimizer_state": {},
    "best_val_acc": None,
    "config": asdict(cfg),
}

torch.save(wrapped, OUT_PATH)
print("Saved wrapped checkpoint to:", OUT_PATH)
