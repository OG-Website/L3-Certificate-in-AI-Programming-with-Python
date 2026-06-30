"""
Unit 4 Assessment 5 - Deep learning training, validation tuning and inference.

This script demonstrates:
- 3.1 training a deep learning model
- 3.2 using a validation set for hyperparameter tuning
- 3.3 using the trained model for inference

It uses a small synthetic dataset so it can run locally without external downloads.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class RunConfig:
    hidden_size: int
    learning_rate: float
    batch_size: int = 32
    epochs: int = 12


class SimpleClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_data_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    x, y = make_classification(
        n_samples=900,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=SEED,
    )

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    def dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    train_loader = DataLoader(dataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    metadata = {
        "input_size": int(x_train.shape[1]),
        "output_size": int(len(np.unique(y))),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "example_for_inference": x_test[0].tolist(),
        "example_label": int(y_test[0]),
    }
    return train_loader, val_loader, test_loader, metadata


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def train_one_config(config: RunConfig, device: torch.device):
    train_loader, val_loader, test_loader, metadata = build_data_loaders(config.batch_size)
    model = SimpleClassifier(
        input_size=metadata["input_size"],
        hidden_size=config.hidden_size,
        output_size=metadata["output_size"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * features.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        row = {
            "hidden_size": config.hidden_size,
            "learning_rate": config.learning_rate,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    return model, metadata, history, {
        "hidden_size": config.hidden_size,
        "learning_rate": config.learning_rate,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_training_plot(rows: list[dict], output_path: Path):
    best_key = min(
        {(r["hidden_size"], r["learning_rate"]) for r in rows},
        key=lambda key: min(
            row["val_loss"]
            for row in rows
            if row["hidden_size"] == key[0] and row["learning_rate"] == key[1]
        ),
    )
    best_rows = [
        row
        for row in rows
        if row["hidden_size"] == best_key[0] and row["learning_rate"] == best_key[1]
    ]
    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in best_rows], [r["train_loss"] for r in best_rows], label="Training loss")
    plt.plot([r["epoch"] for r in best_rows], [r["val_loss"] for r in best_rows], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Assessment 5 training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Assessment 5 test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (row, col), value in np.ndenumerate(matrix):
        plt.text(col, row, str(value), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        RunConfig(hidden_size=32, learning_rate=0.020),
        RunConfig(hidden_size=64, learning_rate=0.010),
        RunConfig(hidden_size=64, learning_rate=0.005),
    ]

    all_rows: list[dict] = []
    results = []
    trained = []
    for config in configs:
        model, metadata, history, result = train_one_config(config, device)
        all_rows.extend(history)
        results.append(result)
        trained.append((model, metadata, result))

    best_index, best_result = min(
        enumerate(results), key=lambda item: item[1]["best_validation_loss"]
    )
    best_model, metadata, _ = trained[best_index]
    checkpoint_path = OUTPUTS / "assessment5_best_model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "metadata": metadata,
            "best_result": {k: v for k, v in best_result.items() if k not in {"y_true", "y_pred"}},
        },
        checkpoint_path,
    )

    metrics_csv = OUTPUTS / "assessment5_training_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    save_training_plot(all_rows, OUTPUTS / "assessment5_training_validation_loss.png")
    save_confusion_matrix(
        best_result["y_true"],
        best_result["y_pred"],
        OUTPUTS / "assessment5_test_confusion_matrix.png",
    )

    loaded = torch.load(checkpoint_path, map_location=device)
    inference_model = SimpleClassifier(
        loaded["metadata"]["input_size"],
        loaded["best_result"]["hidden_size"],
        loaded["metadata"]["output_size"],
    ).to(device)
    inference_model.load_state_dict(loaded["model_state_dict"])
    inference_model.eval()
    sample = torch.tensor([metadata["example_for_inference"]], dtype=torch.float32).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(inference_model(sample), dim=1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probabilities))

    summary = {
        "assessment": "Unit 4 Assessment 5",
        "criteria_covered": ["3.1", "3.2", "3.3"],
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "data_split": {
            "train": metadata["train_samples"],
            "validation": metadata["validation_samples"],
            "test": metadata["test_samples"],
        },
        "hyperparameter_configs_tested": [config.__dict__ for config in configs],
        "best_result": {k: v for k, v in best_result.items() if k not in {"y_true", "y_pred"}},
        "inference": {
            "actual_class": metadata["example_label"],
            "predicted_class": predicted_class,
            "probabilities": probabilities.round(4).tolist(),
        },
        "outputs": [
            str(checkpoint_path.name),
            str(metrics_csv.name),
            "assessment5_training_validation_loss.png",
            "assessment5_test_confusion_matrix.png",
        ],
    }
    (OUTPUTS / "assessment5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
