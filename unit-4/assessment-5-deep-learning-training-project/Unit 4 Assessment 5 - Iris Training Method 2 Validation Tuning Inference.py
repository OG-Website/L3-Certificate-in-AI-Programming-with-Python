"""
Unit 4 Assessment 5 - Iris Training Method 2, validation tuning and inference.

This file demonstrates the second training route for the assessment. It uses a
validation set to tune hidden size and learning rate, then runs inference with
the selected model.

Criteria covered:
- 3.1 Demonstrate a way to train the deep learning model.
- 3.2 Demonstrate the use of a validation set for hyperparameter tuning.
- 3.3 Demonstrate how to use deep learning for inference.

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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


class IrisTunedNet(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def prepare_data():
    iris = load_iris()
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        iris.data.astype(np.float32),
        iris.target.astype(np.int64),
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )
    scaler = StandardScaler()
    return (
        iris,
        scaler.fit_transform(x_train).astype(np.float32),
        scaler.transform(x_val).astype(np.float32),
        scaler.transform(x_test).astype(np.float32),
        y_train,
        y_val,
        y_test,
    )


def train_config(hidden_size: int, learning_rate: float):
    iris, x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    model = IrisTunedNet(hidden_size)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=16,
        shuffle=True,
    )
    x_val_t = torch.tensor(x_val)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_loss = float("inf")
    best_state = None
    for _ in range(50):
        model.train()
        for batch_x, batch_y in data_loader:
            optimiser.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return iris, model, best_loss, x_test, y_test


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    configs = [(8, 0.02), (16, 0.01), (32, 0.005)]
    trained = []
    for hidden_size, learning_rate in configs:
        iris, model, val_loss, x_test, y_test = train_config(hidden_size, learning_rate)
        trained.append((hidden_size, learning_rate, iris, model, val_loss, x_test, y_test))
    selected = min(trained, key=lambda item: item[4])
    hidden_size, learning_rate, iris, model, val_loss, x_test, y_test = selected
    with torch.no_grad():
        logits = model(torch.tensor(x_test))
        predictions = torch.argmax(logits, dim=1).numpy()
        probabilities = torch.softmax(model(torch.tensor(x_test[:1])), dim=1).numpy()[0]
    result = {
        "selected_hidden_size": hidden_size,
        "selected_learning_rate": learning_rate,
        "best_validation_loss": float(val_loss),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "inference_actual": str(iris.target_names[int(y_test[0])]),
        "inference_prediction": str(iris.target_names[int(np.argmax(probabilities))]),
        "inference_probabilities": {str(name): float(probabilities[index]) for index, name in enumerate(iris.target_names)},
    }
    torch.save(model.state_dict(), OUTPUTS / "iris_method_2_tuned_model.pt")
    (OUTPUTS / "iris_method_2_tuned_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
