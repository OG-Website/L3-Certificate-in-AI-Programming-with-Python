"""
Unit 4 Assessment 6 - pre-trained networks, GPU-aware training and image processing.

This script demonstrates:
- 3.3 deep learning inference
- 4.1 use of a pre-trained network
- 4.2 a GPU-aware training script
- 4.3 image processing
- 4.4 data normalisation
- 4.5 data augmentation
- 4.6 data loading and batching for training, validation and testing
- 4.7 saving a model and loading checkpoints

The script uses generated image data so it can run locally without internet access.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS_NAMES = ["vertical", "horizontal", "diagonal"]
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(3, 1, 1)


@dataclass
class TrainResult:
    train_loss: float
    val_loss: float
    val_accuracy: float


class SyntheticShapeDataset(Dataset):
    def __init__(
        self,
        sample_count: int,
        image_size: int = 32,
        augment: bool = False,
        source_task: bool = False,
        labels: list[int] | None = None,
        index_offset: int = 0,
    ):
        self.sample_count = sample_count
        self.image_size = image_size
        self.augment = augment
        self.source_task = source_task
        self.index_offset = index_offset
        if labels is None:
            self.labels = [index % len(CLASS_NAMES) for index in range(sample_count)]
            random.Random(SEED).shuffle(self.labels)
        else:
            self.labels = labels

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int):
        label = self.labels[index]
        image = self._create_image(label, index + self.index_offset)
        if self.augment:
            image = self._augment_image(image)
        tensor = self._to_tensor(image)
        tensor = (tensor - MEAN) / STD
        return tensor, torch.tensor(label, dtype=torch.long)

    def _create_image(self, label: int, index: int) -> Image.Image:
        rng = random.Random(SEED + index + (1000 if self.source_task else 0))
        size = self.image_size
        background = tuple(int(x) for x in rng.choices(range(24, 65), k=3))
        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)
        thickness = rng.randint(4, 7)
        offset = rng.randint(8, 22)
        colour = [(220, 70, 70), (70, 200, 95), (85, 120, 230)][label]

        if label == 0:
            draw.rectangle((offset, 3, offset + thickness, size - 4), fill=colour)
        elif label == 1:
            draw.rectangle((3, offset, size - 4, offset + thickness), fill=colour)
        else:
            draw.line((5, size - 6, size - 6, 5), fill=colour, width=thickness)

        if self.source_task:
            # The source task has the same feature extractor need, but a slightly
            # different style, so the later model can reuse learned edge features.
            draw.ellipse((2, 2, 10, 10), outline=(230, 230, 230), width=2)
        return image

    def _augment_image(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.35:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        angle = random.uniform(-12, 12)
        image = image.rotate(angle)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.85, 1.15))
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.tensor(array, dtype=torch.float32)


class TinyFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ShapeClassifier(nn.Module):
    def __init__(self, feature_extractor: TinyFeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(24 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)


def create_loaders(batch_size: int = 32, source_task: bool = False):
    labels = [index % len(CLASS_NAMES) for index in range(540)]
    random.Random(SEED + (10 if source_task else 0)).shuffle(labels)
    train_labels = labels[:360]
    val_labels = labels[360:450]
    test_labels = labels[450:]
    train_dataset = SyntheticShapeDataset(
        len(train_labels),
        augment=True,
        source_task=source_task,
        labels=train_labels,
        index_offset=0,
    )
    val_dataset = SyntheticShapeDataset(
        len(val_labels),
        augment=False,
        source_task=source_task,
        labels=val_labels,
        index_offset=1000,
    )
    test_dataset = SyntheticShapeDataset(
        len(test_labels),
        augment=False,
        source_task=source_task,
        labels=test_labels,
        index_offset=2000,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "mean": MEAN.flatten().tolist(),
                    "std": STD.flatten().tolist(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
    return history


def save_metrics(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_loss_plot(rows: list[dict], path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label="Training loss")
    plt.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Assessment 6 image model training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion(y_true: list[int], y_pred: list[int], path: Path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Purples")
    plt.title("Assessment 6 test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (row, col), value in np.ndenumerate(matrix):
        plt.text(col, row, str(value), ha="center", va="center")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_inference_image(dataset: SyntheticShapeDataset, model: nn.Module, device: torch.device, path: Path):
    raw_image = dataset._create_image(label=2, index=999)
    tensor = dataset._to_tensor(raw_image)
    normalised = ((tensor - MEAN) / STD).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(normalised), dim=1).cpu().numpy()[0]
    predicted = int(np.argmax(probabilities))
    scale = 8
    large = raw_image.resize((raw_image.width * scale, raw_image.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (large.width, large.height + 42), "white")
    canvas.paste(large, (0, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)
    draw.rectangle((0, 0, canvas.width, 40), fill=(0, 0, 0))
    draw.text((8, 9), f"Predicted: {CLASS_NAMES[predicted]}", fill=(255, 255, 255), font=font)
    canvas.save(path)
    return {
        "expected_shape": CLASS_NAMES[2],
        "predicted_shape": CLASS_NAMES[predicted],
        "probabilities": probabilities.round(4).tolist(),
    }


def save_preprocessing_preview(path: Path):
    dataset = SyntheticShapeDataset(1, augment=True)
    raw = dataset._create_image(label=0, index=123)
    augmented = dataset._augment_image(raw)
    tensor = dataset._to_tensor(augmented)
    normalised = (tensor - MEAN) / STD
    scale = 5
    tile_size = 32 * scale
    preview = Image.new("RGB", (tile_size * 3, tile_size + 34), "white")
    font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 16)
    preview.paste(raw.resize((tile_size, tile_size), Image.Resampling.NEAREST), (0, 34))
    preview.paste(augmented.resize((tile_size, tile_size), Image.Resampling.NEAREST), (tile_size, 34))
    restored = torch.clamp((normalised * STD) + MEAN, 0, 1)
    restored_array = (restored.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    restored_img = Image.fromarray(restored_array).resize((tile_size, tile_size), Image.Resampling.NEAREST)
    preview.paste(restored_img, (tile_size * 2, 34))
    draw = ImageDraw.Draw(preview)
    draw.text((8, 8), "Raw", fill=(0, 0, 0), font=font)
    draw.text((tile_size + 8, 8), "Augmented", fill=(0, 0, 0), font=font)
    draw.text((tile_size * 2 + 8, 8), "Normalised/restored", fill=(0, 0, 0), font=font)
    preview.save(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    source_train, source_val, _ = create_loaders(source_task=True)
    pretrain_model = ShapeClassifier(TinyFeatureExtractor()).to(device)
    pretrain_checkpoint = OUTPUTS / "assessment6_pretrained_feature_checkpoint.pt"
    pretrain_history = train_model(
        pretrain_model,
        source_train,
        source_val,
        device,
        epochs=3,
        learning_rate=0.003,
        checkpoint_path=pretrain_checkpoint,
    )

    loaded_pretrain = torch.load(pretrain_checkpoint, map_location=device)
    feature_extractor = TinyFeatureExtractor()
    temp_model = ShapeClassifier(feature_extractor)
    temp_model.load_state_dict(loaded_pretrain["model_state_dict"])

    target_model = ShapeClassifier(temp_model.feature_extractor).to(device)
    train_loader, val_loader, test_loader = create_loaders(source_task=False)
    target_checkpoint = OUTPUTS / "assessment6_best_model_checkpoint.pt"
    target_history = train_model(
        target_model,
        train_loader,
        val_loader,
        device,
        epochs=6,
        learning_rate=0.002,
        checkpoint_path=target_checkpoint,
    )

    loaded_target = torch.load(target_checkpoint, map_location=device)
    target_model.load_state_dict(loaded_target["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, y_true, y_pred = evaluate(target_model, test_loader, criterion, device)

    save_metrics(OUTPUTS / "assessment6_pretrain_metrics.csv", pretrain_history)
    save_metrics(OUTPUTS / "assessment6_training_metrics.csv", target_history)
    save_loss_plot(target_history, OUTPUTS / "assessment6_training_validation_loss.png")
    save_confusion(y_true, y_pred, OUTPUTS / "assessment6_test_confusion_matrix.png")
    save_preprocessing_preview(OUTPUTS / "assessment6_image_processing_preview.png")
    inference = save_inference_image(
        SyntheticShapeDataset(1, augment=False),
        target_model,
        device,
        OUTPUTS / "assessment6_inference_result.png",
    )

    summary = {
        "assessment": "Unit 4 Assessment 6",
        "criteria_covered": ["3.3", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7"],
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "pretrained_network": "TinyFeatureExtractor pre-trained on source synthetic image task and loaded into target ShapeClassifier",
        "data_loading_and_batching": {
            "train_batches": len(train_loader),
            "validation_batches": len(val_loader),
            "test_batches": len(test_loader),
            "batch_size": train_loader.batch_size,
        },
        "image_processing": [
            "PIL image generation",
            "RGB conversion into tensor format",
            "random flips",
            "random rotation",
            "brightness augmentation",
            "normalisation with mean and standard deviation",
        ],
        "checkpoint_files": [pretrain_checkpoint.name, target_checkpoint.name],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "inference": inference,
        "outputs": [
            "assessment6_pretrain_metrics.csv",
            "assessment6_training_metrics.csv",
            "assessment6_training_validation_loss.png",
            "assessment6_test_confusion_matrix.png",
            "assessment6_image_processing_preview.png",
            "assessment6_inference_result.png",
        ],
    }
    (OUTPUTS / "assessment6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
