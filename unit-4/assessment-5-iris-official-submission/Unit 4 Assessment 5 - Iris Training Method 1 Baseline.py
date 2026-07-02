"""
Unit 4 Assessment 5 - Iris Training Method 1 Baseline.

This file demonstrates the first Iris-based way to train a deep learning model.
It uses the Iris dataset supplied through sklearn.datasets.load_iris.

Criteria covered:
- 3.1 Demonstrate a way to train the deep learning model.

Dataset source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


class IrisBaselineNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data.astype(np.float32),
        iris.target.astype(np.int64),
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=16,
        shuffle=True,
    )
    model = IrisBaselineNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.CrossEntropyLoss()
    for _ in range(40):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        predictions = torch.argmax(model(torch.tensor(x_test)), dim=1).numpy()
    accuracy = accuracy_score(y_test, predictions)
    torch.save(model.state_dict(), OUTPUTS / "iris_method_1_baseline_model.pt")
    (OUTPUTS / "iris_method_1_baseline_results.json").write_text(
        json.dumps({"dataset": "Iris", "method": "baseline", "test_accuracy": float(accuracy)}, indent=2),
        encoding="utf-8",
    )
    print(f"Iris baseline test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
