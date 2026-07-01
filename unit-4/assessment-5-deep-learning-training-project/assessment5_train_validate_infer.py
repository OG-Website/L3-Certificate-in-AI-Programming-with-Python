"""
Unit 4 Assessment 5 - deep learning training, validation tuning and inference.

This script generates the evidence for:
- 3.1 two ways to train a deep learning model: Iris data and synthetic data
- 3.2 validation-set hyperparameter tuning using the Iris data
- 3.3 inference using the trained Iris model

It runs locally without external downloads.
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
from sklearn.datasets import load_iris, make_classification
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
    batch_size: int = 16
    epochs: int = 35


class Classifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    test_size: float = 0.20,
    validation_size: float = 0.20,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_size + validation_size, random_state=SEED, stratify=y
    )
    relative_test_size = test_size / (test_size + validation_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=y_temp,
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
        "scaler_mean": scaler.mean_.round(6).tolist(),
        "scaler_scale": scaler.scale_.round(6).tolist(),
        "example_for_inference": x_test[0].tolist(),
        "example_label": int(y_test[0]),
    }
    return train_loader, val_loader, test_loader, metadata


def iris_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    iris = load_iris()
    return iris.data.astype(np.float32), iris.target.astype(np.int64), [str(name) for name in iris.target_names]


def synthetic_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    x, y = make_classification(
        n_samples=900,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.6,
        random_state=SEED,
    )
    return x.astype(np.float32), y.astype(np.int64), ["class 0", "class 1", "class 2"]


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


def train_dataset(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    configs: list[RunConfig],
    device: torch.device,
):
    all_rows: list[dict] = []
    results = []
    trained = []
    for config in configs:
        train_loader, val_loader, test_loader, metadata = make_loaders(x, y, config.batch_size)
        model = Classifier(metadata["input_size"], config.hidden_size, metadata["output_size"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        best_state = None
        best_val_loss = float("inf")

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
            all_rows.append(
                {
                    "dataset": name,
                    "hidden_size": config.hidden_size,
                    "learning_rate": config.learning_rate,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        assert best_state is not None
        model.load_state_dict(best_state)
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        result = {
            "dataset": name,
            "hidden_size": config.hidden_size,
            "learning_rate": config.learning_rate,
            "best_validation_loss": best_val_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "y_true": y_true,
            "y_pred": y_pred,
        }
        results.append(result)
        trained.append((model, metadata, result, class_names))

    best_index, best_result = min(enumerate(results), key=lambda item: item[1]["best_validation_loss"])
    best_model, metadata, _, class_names = trained[best_index]
    return best_model, metadata, best_result, class_names, all_rows, results


def save_rows(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_result(result: dict) -> dict:
    return {
        "dataset": result["dataset"],
        "hidden_size": int(result["hidden_size"]),
        "learning_rate": float(result["learning_rate"]),
        "best_validation_loss": float(result["best_validation_loss"]),
        "test_loss": float(result["test_loss"]),
        "test_accuracy": float(result["test_accuracy"]),
    }


def save_loss_plot(rows: list[dict], dataset: str, output_path: Path):
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    best_key = min(
        {(row["hidden_size"], row["learning_rate"]) for row in dataset_rows},
        key=lambda key: min(
            row["val_loss"]
            for row in dataset_rows
            if row["hidden_size"] == key[0] and row["learning_rate"] == key[1]
        ),
    )
    best_rows = [
        row
        for row in dataset_rows
        if row["hidden_size"] == best_key[0] and row["learning_rate"] == best_key[1]
    ]
    plt.figure(figsize=(8, 5))
    plt.plot([r["epoch"] for r in best_rows], [r["train_loss"] for r in best_rows], label="Training loss")
    plt.plot([r["epoch"] for r in best_rows], [r["val_loss"] for r in best_rows], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Assessment 5 {dataset} training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_tuning_summary(results: list[dict], output_path: Path):
    rows = [
        [
            result["dataset"],
            f"{result['hidden_size']}",
            f"{result['learning_rate']:.3f}",
            f"{result['best_validation_loss']:.4f}",
            f"{result['test_accuracy']:.2%}",
        ]
        for result in results
    ]
    plt.figure(figsize=(9, 2.8))
    plt.axis("off")
    table = plt.table(
        cellText=rows,
        colLabels=["Dataset", "Hidden size", "Learning rate", "Best val loss", "Test accuracy"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    plt.title("Assessment 5 Iris validation tuning", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_confusion(y_true: list[int], y_pred: list[int], class_names: list[str], output_path: Path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.4, 4.6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Assessment 5 Iris test confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=25)
    plt.yticks(range(len(class_names)), class_names)
    for (row, col), value in np.ndenumerate(matrix):
        plt.text(col, row, str(value), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_inference(summary: dict, output_path: Path):
    inference = summary["iris_inference"]
    probabilities = inference["probabilities"]
    labels = inference["class_names"]
    plt.figure(figsize=(7.2, 4))
    bars = plt.bar(labels, probabilities, color=["#4c78a8", "#59a14f", "#f28e2b"])
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title(
        f"Iris inference: actual {inference['actual_class_name']} / "
        f"predicted {inference['predicted_class_name']}"
    )
    for bar, value in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iris_x, iris_y, iris_names = iris_arrays()
    synthetic_x, synthetic_y, synthetic_names = synthetic_arrays()

    iris_configs = [
        RunConfig(hidden_size=8, learning_rate=0.020, epochs=40),
        RunConfig(hidden_size=16, learning_rate=0.010, epochs=40),
        RunConfig(hidden_size=32, learning_rate=0.005, epochs=40),
    ]
    synthetic_configs = [RunConfig(hidden_size=48, learning_rate=0.010, batch_size=32, epochs=18)]

    iris_model, iris_metadata, iris_best, _, iris_rows, iris_results = train_dataset(
        "iris", iris_x, iris_y, iris_names, iris_configs, device
    )
    synthetic_model, synthetic_metadata, synthetic_best, _, synthetic_rows, _ = train_dataset(
        "synthetic", synthetic_x, synthetic_y, synthetic_names, synthetic_configs, device
    )
    all_rows = iris_rows + synthetic_rows
    save_rows(OUTPUTS / "assessment5_training_metrics.csv", all_rows)

    iris_checkpoint = OUTPUTS / "assessment5_iris_best_model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": iris_model.state_dict(),
            "metadata": iris_metadata,
            "class_names": iris_names,
            "best_result": checkpoint_result(iris_best),
        },
        iris_checkpoint,
    )
    synthetic_checkpoint = OUTPUTS / "assessment5_synthetic_model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": synthetic_model.state_dict(),
            "metadata": synthetic_metadata,
            "class_names": synthetic_names,
            "best_result": checkpoint_result(synthetic_best),
        },
        synthetic_checkpoint,
    )

    save_loss_plot(all_rows, "iris", OUTPUTS / "assessment5_iris_training_loss.png")
    save_loss_plot(all_rows, "synthetic", OUTPUTS / "assessment5_synthetic_training_loss.png")
    save_tuning_summary(iris_results, OUTPUTS / "assessment5_iris_validation_tuning_summary.png")
    save_confusion(
        iris_best["y_true"],
        iris_best["y_pred"],
        iris_names,
        OUTPUTS / "assessment5_iris_test_confusion_matrix.png",
    )

    loaded = torch.load(iris_checkpoint, map_location=device)
    inference_model = Classifier(
        loaded["metadata"]["input_size"],
        loaded["best_result"]["hidden_size"],
        loaded["metadata"]["output_size"],
    ).to(device)
    inference_model.load_state_dict(loaded["model_state_dict"])
    inference_model.eval()
    sample = torch.tensor([loaded["metadata"]["example_for_inference"]], dtype=torch.float32).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(inference_model(sample), dim=1).cpu().numpy()[0]
    predicted_class = int(np.argmax(probabilities))
    actual_class = int(loaded["metadata"]["example_label"])

    summary = {
        "assessment": "Unit 4 Assessment 5",
        "criteria_covered": ["3.1", "3.2", "3.3"],
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "training_demonstrations": {
            "iris": {
                "train": iris_metadata["train_samples"],
                "validation": iris_metadata["validation_samples"],
                "test": iris_metadata["test_samples"],
                "best_validation_loss": iris_best["best_validation_loss"],
                "test_accuracy": iris_best["test_accuracy"],
            },
            "synthetic": {
                "train": synthetic_metadata["train_samples"],
                "validation": synthetic_metadata["validation_samples"],
                "test": synthetic_metadata["test_samples"],
                "best_validation_loss": synthetic_best["best_validation_loss"],
                "test_accuracy": synthetic_best["test_accuracy"],
            },
        },
        "iris_hyperparameter_configs_tested": [config.__dict__ for config in iris_configs],
        "iris_best_result": {k: v for k, v in iris_best.items() if k not in {"y_true", "y_pred"}},
        "iris_inference": {
            "actual_class": actual_class,
            "actual_class_name": iris_names[actual_class],
            "predicted_class": predicted_class,
            "predicted_class_name": iris_names[predicted_class],
            "class_names": iris_names,
            "probabilities": probabilities.round(4).tolist(),
        },
        "outputs": [
            "assessment5_iris_best_model_checkpoint.pt",
            "assessment5_synthetic_model_checkpoint.pt",
            "assessment5_training_metrics.csv",
            "assessment5_summary.json",
            "assessment5_iris_training_loss.png",
            "assessment5_synthetic_training_loss.png",
            "assessment5_iris_validation_tuning_summary.png",
            "assessment5_iris_test_confusion_matrix.png",
            "assessment5_iris_inference_result.png",
        ],
    }
    save_inference(summary, OUTPUTS / "assessment5_iris_inference_result.png")
    (OUTPUTS / "assessment5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
