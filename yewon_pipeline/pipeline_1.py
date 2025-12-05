import os
import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms, models


@dataclass
class TrainConfig:
 
    data_root: str
    output_dir: str = "./runs/exp1"

    # Dataset mode
    dataset_mode: str = "cropped"   # "cropped" or "mask12k"

    num_classes: int = 2                # 2 or 3
    merge_incorrect_with_nomask: bool = True  # Whether to merge incorrect with no-mask when num_classes==2

    # Model
    model_name: str = "resnet18"        # See build_model() below for options
    pretrained: bool = True
    freeze_backbone: bool = True

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4

    img_size: int = 224
    seed: int = 42

    # Distributed training (torchrun sets via env)
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    # mixed precision
    amp: bool = True



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(cfg: TrainConfig) -> bool:
    return (not cfg.distributed) or cfg.rank == 0


def init_distributed_mode(cfg: TrainConfig):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cfg.distributed = cfg.world_size > 1
    else:
        cfg.rank = 0
        cfg.world_size = 1
        cfg.local_rank = 0
        cfg.distributed = False

    if cfg.distributed:
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(
            backend=cfg.backend,
            init_method="env://",
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(
    cfg: TrainConfig,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_val_acc: float,
    filename: str = "best.pth",
):
    if not is_main_process(cfg):
        return
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, filename)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "config": asdict(cfg),
    }
    torch.save(state, path)


def parse_kaggle_annotation(xml_path: str) -> Tuple[List[List[int]], List[str]]:
    """
    Parse Kaggle face-mask XML annotation:
      boxes: [[xmin, ymin, xmax, ymax], ...]
      labels: ["with_mask", "without_mask", "mask_weared_incorrect", ...]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes: List[List[int]] = []
    labels: List[str] = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)

    return boxes, labels


def map_face_label(
    raw_label: str,
    num_classes: int = 2,
    merge_incorrect_with_nomask: bool = True,
) -> Optional[int]:

    if num_classes == 2:
        if raw_label == "with_mask":
            return 1
        elif raw_label in ["without_mask", "mask_weared_incorrect"]:
            if merge_incorrect_with_nomask:
                return 0
            else:
                return 0 if raw_label == "without_mask" else None
        else:
            return None
    elif num_classes == 3:
        if raw_label == "with_mask":
            return 0
        elif raw_label == "without_mask":
            return 1
        elif raw_label == "mask_weared_incorrect":
            return 2
        else:
            return None
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


class CroppedFaceMaskDataset(Dataset):
    """
    Dataset that uses face crops from bounding boxes as samples.
    """

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        num_classes: int = 2,
        merge_incorrect_with_nomask: bool = True,
        transform: Optional[transforms.Compose] = None,
        max_faces_per_image: Optional[int] = None,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.num_classes = num_classes
        self.merge_incorrect_with_nomask = merge_incorrect_with_nomask
        self.transform = transform
        self.max_faces_per_image = max_faces_per_image

        self.samples: List[Tuple[str, List[int], int]] = []

        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
        xml_files.sort()

        for xml_name in xml_files:
            xml_path = os.path.join(annotations_dir, xml_name)
            boxes, raw_labels = parse_kaggle_annotation(xml_path)

            base_name = os.path.splitext(xml_name)[0] + ".png"
            img_path = os.path.join(images_dir, base_name)
            if not os.path.exists(img_path):
                base_name = os.path.splitext(xml_name)[0] + ".jpg"
                img_path = os.path.join(images_dir, base_name)
            if not os.path.exists(img_path):
                continue

            per_image_faces = 0
            for box, raw_label in zip(boxes, raw_labels):
                mapped = map_face_label(
                    raw_label,
                    num_classes=self.num_classes,
                    merge_incorrect_with_nomask=self.merge_incorrect_with_nomask,
                )
                if mapped is None:
                    continue

                self.samples.append((img_path, box, mapped))
                per_image_faces += 1
                if (
                    self.max_faces_per_image is not None
                    and per_image_faces >= self.max_faces_per_image
                ):
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, box, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        xmin, ymin, xmax, ymax = box
        img = img.crop((xmin, ymin, xmax, ymax))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class Mask12KFolderDataset(Dataset):
    """
    For Face Mask 12K Dataset.
    split_dir:
        <data_root>/Train
        <data_root>/Validation
        <data_root>/Test
    Each contains WithMask / WithoutMask folders.
    Labels: 0 = WithoutMask, 1 = WithMask.
    """
    def __init__(self, split_dir: str,
                 transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform

        class_specs = [("WithMask", 1), ("WithoutMask", 0)]
        for class_name, label in class_specs:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                path = os.path.join(class_dir, fname)
                self.samples.append((path, label))

        # Sorting helps with reproducibility
        self.samples.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(img_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def build_datasets_and_loaders(cfg: TrainConfig):
    train_transform = build_transforms(cfg.img_size, is_train=True)
    val_transform = build_transforms(cfg.img_size, is_train=False)

    if cfg.dataset_mode == "mask12k" and cfg.num_classes != 2:
        raise ValueError("Mask12K dataset only supports num_classes=2.")
    
    if cfg.dataset_mode == "cropped":
        # === andrewmvd dataset (cropped) ===
        images_dir = os.path.join(cfg.data_root, "images")
        annotations_dir = os.path.join(cfg.data_root, "annotations")

        full_dataset = CroppedFaceMaskDataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            num_classes=cfg.num_classes,
            merge_incorrect_with_nomask=cfg.merge_incorrect_with_nomask,
            transform=train_transform,
        )
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

        full_dataset_val = CroppedFaceMaskDataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            num_classes=cfg.num_classes,
            merge_incorrect_with_nomask=cfg.merge_incorrect_with_nomask,
            transform=val_transform,
        )
        val_dataset = torch.utils.data.Subset(full_dataset_val, val_indices)

    elif cfg.dataset_mode == "mask12k":
        # === Face Mask 12K (classification only) ===
        train_dir = os.path.join(cfg.data_root, "Train")
        val_dir = os.path.join(cfg.data_root, "Validation")

        train_dataset = Mask12KFolderDataset(
            split_dir=train_dir,
            transform=train_transform,
        )
        val_dataset = Mask12KFolderDataset(
            split_dir=val_dir,
            transform=val_transform,
        )
    else:
        raise ValueError(f"Unknown dataset_mode: {cfg.dataset_mode}")

    # Below is common (DDP & DataLoader)
    if cfg.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader



class SmallCNN(nn.Module):
    """
    Simple baseline CNN (RGB, 224x224).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# custom
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CustomCNN(nn.Module):
    """
    Custom depthwise-separable CNN architecture.
    Input resolution is flexible due to AdaptiveAvgPool (40x40, 96x96, 224x224 all supported).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # stem
        self.conv_stem = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # depthwise-separable blocks
        self.depthwise_block1 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.depthwise_block2 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # global pooling + head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.depthwise_block1(x)
        x = self.pool2(x)
        x = self.depthwise_block2(x)
        x = self.pool3(x)

        x = self.global_avg_pool(x)      # (N, 128, 1, 1)
        x = torch.flatten(x, 1)          # (N, 128)
        x = self.dropout(x)
        x = self.fc(x)                   # (N, num_classes)
        return x



def build_model(cfg: TrainConfig) -> nn.Module:
    """
    Create classifier backbone + head.
    """
    name = cfg.model_name.lower()

    if name == "resnet18":
        model = models.resnet18(pretrained=cfg.pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, cfg.num_classes),
        )
        if cfg.freeze_backbone:
            for n, p in model.named_parameters():
                if not n.startswith("fc"):
                    p.requires_grad = False

    elif name == "resnet50":
        model = models.resnet50(pretrained=cfg.pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.num_classes),
        )
        if cfg.freeze_backbone:
            for n, p in model.named_parameters():
                if not n.startswith("fc"):
                    p.requires_grad = False

    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=cfg.pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, cfg.num_classes)
        if cfg.freeze_backbone:
            for n, p in model.named_parameters():
                if "classifier" not in n:
                    p.requires_grad = False

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=cfg.pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, cfg.num_classes)
        if cfg.freeze_backbone:
            for n, p in model.named_parameters():
                if "classifier" not in n:
                    p.requires_grad = False

    elif name == "vit_b_16":
        model = models.vit_b_16(pretrained=cfg.pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, cfg.num_classes)
        if cfg.freeze_backbone:
            for n, p in model.named_parameters():
                if "heads.head" not in n:
                    p.requires_grad = False

    elif name == "small_cnn":
        model = SmallCNN(num_classes=cfg.num_classes)

    elif name == "custom_cnn":
        model = CustomCNN(in_channels=3, num_classes=cfg.num_classes)

    else:
        raise ValueError(f"Unknown model_name: {cfg.model_name}")

    return model


def build_optimizer(cfg: TrainConfig, model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


# ================================
# Train / Eval
# ================================

def reduce_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def train_one_epoch(
    cfg: TrainConfig,
    epoch: int,
    model: nn.Module,
    criterion,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train()
    if cfg.distributed:
        assert isinstance(data_loader.sampler, DistributedSampler)
        data_loader.sampler.set_epoch(epoch)

    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if cfg.amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    device_tensor = torch.tensor(
        [loss_sum, correct, total], dtype=torch.float64, device=device
    )
    device_tensor = reduce_across_processes(device_tensor)
    loss_sum, correct, total = device_tensor.tolist()

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)

    return avg_loss, acc


@torch.no_grad()
def evaluate(
    cfg: TrainConfig,
    model: nn.Module,
    criterion,
    data_loader: DataLoader,
    device: torch.device,
):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    device_tensor = torch.tensor(
        [loss_sum, correct, total], dtype=torch.float64, device=device
    )
    device_tensor = reduce_across_processes(device_tensor)
    loss_sum, correct, total = device_tensor.tolist()

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)

    return avg_loss, acc


def run_train(cfg: TrainConfig):
    init_distributed_mode(cfg)
    set_seed(cfg.seed)

    device = torch.device(
        f"cuda:{cfg.local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if is_main_process(cfg):
        os.makedirs(cfg.output_dir, exist_ok=True)
        print("======== TrainConfig ========")
        for k, v in asdict(cfg).items():
            print(f"{k}: {v}")
        print("=============================")

    train_loader, val_loader = build_datasets_and_loaders(cfg)

    model = build_model(cfg)
    model.to(device)

    if cfg.distributed:
        model = DDP(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        if is_main_process(cfg):
            print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")

        train_loss, train_acc = train_one_epoch(
            cfg, epoch, model, criterion, optimizer, train_loader, device, scaler
        )
        val_loss, val_acc = evaluate(cfg, model, criterion, val_loader, device)

        if is_main_process(cfg):
            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # Save best model
        if val_acc > best_val_acc and is_main_process(cfg):
            best_val_acc = val_acc
            save_checkpoint(
                cfg,
                epoch,
                model.module if isinstance(model, DDP) else model,
                optimizer,
                best_val_acc,
                filename="best.pth",
            )

    if is_main_process(cfg):
        print(f"Training finished. Best val_acc={best_val_acc:.4f}")
    cleanup_distributed()



def build_argparser():
    p = argparse.ArgumentParser(
        description="Face Mask Detection Training Pipeline (PyTorch, multi-GPU ready)"
    )
    p.add_argument(
        "--dataset_mode",
        type=str,
        default="cropped",
        choices=["cropped", "mask12k"],
        help=(
            "'cropped': andrewmvd face-mask-detection.\n"
            "'mask12k': Face Mask 12K (WithMask/WithoutMask in Train/Validation/Test)."
        ),
    )


    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Dataset root. "
            "If dataset_mode='cropped': contains 'images/' and 'annotations/'. "
            "If dataset_mode='mask12k': contains 'Train/', 'Validation/', 'Test/' "
            "with 'WithMask/' and 'WithoutMask/' subfolders."
        ),
    )



    p.add_argument("--output_dir", type=str, default="./runs/exp1")

    p.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    p.add_argument(
        "--no_merge_incorrect",
        action="store_true",
        help="When num_classes==2, do not merge 'mask_weared_incorrect' into no-mask (ignore it).",
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet50",
            "mobilenet_v3_small",
            "efficientnet_b0",
            "vit_b_16",
            "small_cnn",
            "custom_cnn", 
        ],
    )
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--no_freeze_backbone", action="store_true")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training (recommended on modern GPUs).",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision training.",
    )

    return p


def parse_args_to_config() -> TrainConfig:
    parser = build_argparser()
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_mode=args.dataset_mode,
        num_classes=args.num_classes,
        merge_incorrect_with_nomask=not args.no_merge_incorrect,
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        amp=True,
    )

    if args.pretrained and args.no_pretrained:
        raise ValueError("Use only one of --pretrained / --no_pretrained.")
    if args.no_pretrained:
        cfg.pretrained = False
    if args.pretrained:
        cfg.pretrained = True

    if args.freeze_backbone and args.no_freeze_backbone:
        raise ValueError("Use only one of --freeze_backbone / --no_freeze_backbone.")
    if args.freeze_backbone:
        cfg.freeze_backbone = True
    if args.no_freeze_backbone:
        cfg.freeze_backbone = False

    if args.amp and args.no_amp:
        raise ValueError("Use only one of --amp / --no_amp.")
    if args.amp:
        cfg.amp = True
    if args.no_amp:
        cfg.amp = False

    return cfg


def main():
    cfg = parse_args_to_config()
    run_train(cfg)


if __name__ == "__main__":
    main()
