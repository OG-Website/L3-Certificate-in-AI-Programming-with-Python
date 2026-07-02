"""
Unit 4 Assessment 5 - Trade-A-Them-style synthetic deep learning training.

Assessment criteria covered:
- 3.1 Demonstrate a second way to train a deep learning model using a synthetic dataset.

Synthetic data source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

This script does not use live exchange credentials, private account values or real trading records.
The synthetic data is used as safe project-context evidence for Trade-A-Them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

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


def set_seed(seed: int = 84) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingResult:
    dataset: str
    hidden_size: int
    learning_rate: float
    validation_loss: float
    test_accuracy: float
    model_path: str


class SyntheticSignalNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_tradeathem_style_dataset():
    x, y = make_classification(
        n_samples=900,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.55,
        flip_y=0.03,
        random_state=84,
    )
    feature_names = [
        "recent_return",
        "volatility",
        "volume_pressure",
        "momentum_score",
        "trend_strength",
        "drawdown_pressure",
        "spread_signal",
        "order_flow_proxy",
        "risk_score",
        "sentiment_proxy",
    ]
    target_names = ["risk_exit", "hold", "buy_signal"]
    return x, y, feature_names, target_names


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_data():
    x, y, feature_names, target_names = create_tradeathem_style_dataset()
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=84,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=84,
        stratify=y_train_full,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test, feature_names, target_names


def train_synthetic_model(hidden_size: int = 32, learning_rate: float = 0.01, epochs: int = 35) -> tuple[SyntheticSignalNet, float]:
    x_train, x_val, x_test, y_train, y_val, y_test, feature_names, target_names = prepare_data()
    model = SyntheticSignalNet(input_size=x_train.shape[1], hidden_size=hidden_size, output_size=len(target_names))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(x_train, y_train, batch_size=32, shuffle=True)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    best_validation_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(x_val_tensor), y_val_tensor).item()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_validation_loss


def evaluate(model: SyntheticSignalNet, validation_loss: float) -> TrainingResult:
    x_train, x_val, x_test, y_train, y_val, y_test, feature_names, target_names = prepare_data()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        predictions = torch.argmax(logits, dim=1).numpy()
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).tolist()
    model_path = OUTPUTS / "u4a5_tradeathem_synthetic_signal_model.pt"
    torch.save(model.state_dict(), model_path)
    (OUTPUTS / "u4a5_tradeathem_synthetic_results.json").write_text(
        json.dumps(
            {
                "project_context": "Trade-A-Them active cryptocurrency trading bot",
                "dataset_type": "synthetic classification data",
                "feature_names": feature_names,
                "target_names": target_names,
                "validation_loss": validation_loss,
                "test_accuracy": accuracy,
                "confusion_matrix": matrix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingResult(
        dataset="Trade-A-Them-style synthetic cryptocurrency signal dataset",
        hidden_size=32,
        learning_rate=0.01,
        validation_loss=validation_loss,
        test_accuracy=accuracy,
        model_path=str(model_path),
    )


def main() -> None:
    set_seed()
    model, validation_loss = train_synthetic_model()
    result = evaluate(model, validation_loss)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
