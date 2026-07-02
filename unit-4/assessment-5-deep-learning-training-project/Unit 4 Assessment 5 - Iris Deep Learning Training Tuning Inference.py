"""
Unit 4 Assessment 5 - Iris deep learning training, validation tuning and inference.

Assessment criteria covered:
- 3.1 Demonstrate one way to train a deep learning model using the Iris dataset.
- 3.2 Demonstrate the use of a validation set for hyperparameter tuning.
- 3.3 Demonstrate how to use deep learning for inference.

Dataset source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class ModelResult:
    hidden_size: int
    learning_rate: float
    best_validation_loss: float
    test_accuracy: float
    model_path: str


class IrisNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_data():
    iris = load_iris()
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        iris.data,
        iris.target,
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
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return iris, x_train, x_val, x_test, y_train, y_val, y_test


def train_model(hidden_size: int, learning_rate: float, epochs: int = 40) -> tuple[IrisNet, float]:
    iris, x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    model = IrisNet(input_size=x_train.shape[1], hidden_size=hidden_size, output_size=len(iris.target_names))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(x_train, y_train, batch_size=16, shuffle=True)
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


def evaluate_model(model: IrisNet, hidden_size: int, learning_rate: float, best_validation_loss: float) -> ModelResult:
    iris, x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
        predictions = torch.argmax(logits, dim=1).numpy()
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions).tolist()
    model_path = OUTPUTS / "u4a5_iris_model.pt"
    torch.save(model.state_dict(), model_path)
    (OUTPUTS / "u4a5_iris_results.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "best_validation_loss": best_validation_loss,
                "test_accuracy": accuracy,
                "confusion_matrix": matrix,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ModelResult(hidden_size, learning_rate, best_validation_loss, accuracy, str(model_path))


def run_inference(model: IrisNet) -> None:
    iris, x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()
    sample = torch.tensor(x_test[:1], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(sample), dim=1).numpy()[0]
    prediction_index = int(np.argmax(probabilities))
    inference = {
        "actual_class": iris.target_names[int(y_test[0])],
        "predicted_class": iris.target_names[prediction_index],
        "probabilities": {
            iris.target_names[index]: float(value)
            for index, value in enumerate(probabilities)
        },
    }
    (OUTPUTS / "u4a5_iris_inference.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")


def main() -> None:
    set_seed()
    tuning_grid = [
        {"hidden_size": 8, "learning_rate": 0.02},
        {"hidden_size": 16, "learning_rate": 0.01},
        {"hidden_size": 32, "learning_rate": 0.005},
    ]
    trained = []
    for config in tuning_grid:
        model, validation_loss = train_model(**config)
        trained.append((config, model, validation_loss))

    best_config, best_model, best_validation_loss = min(trained, key=lambda item: item[2])
    result = evaluate_model(best_model, best_config["hidden_size"], best_config["learning_rate"], best_validation_loss)
    run_inference(best_model)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
