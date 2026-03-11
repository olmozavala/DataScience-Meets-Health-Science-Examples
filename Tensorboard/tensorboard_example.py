"""Train a small neural network to approximate a harmonic function and log
everything to TensorBoard (scalars, graph, best-model checkpoint)."""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def harmonic_function(x: np.ndarray) -> np.ndarray:
    """Target function: a sum of sine harmonics.

    Parameters
    ----------
    x : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        f(x) = sin(x) + 0.5*sin(3x) + 0.25*sin(5x)
    """
    return np.sin(x) + 0.5 * np.sin(3 * x) + 0.25 * np.sin(5 * x)


class HarmonicNet(nn.Module):
    """Simple feed-forward network with two hidden layers."""

    def __init__(self, hidden_size: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch, 1).
        """
        return self.net(x)


def build_datasets(
    n_train: int = 1000, n_val: int = 200
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """
    rng = np.random.default_rng(42)

    x_train = rng.uniform(-2 * np.pi, 2 * np.pi, size=(n_train, 1)).astype(np.float32)
    y_train = harmonic_function(x_train).astype(np.float32)

    x_val = rng.uniform(-2 * np.pi, 2 * np.pi, size=(n_val, 1)).astype(np.float32)
    y_val = harmonic_function(x_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=len(x_val),
    )
    return train_loader, val_loader


def train(
    epochs: int = 200,
    lr: float = 1e-2,
    hidden_size: int = 10,
    log_dir: str = "runs/harmonic",
    best_model_path: str = "best_model.pt",
) -> None:
    """Train the network and log metrics / graph to TensorBoard.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    hidden_size : int
        Neurons per hidden layer.
    log_dir : str
        TensorBoard log directory.
    best_model_path : str
        Path where the best model checkpoint is saved.
    """
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    train_loader, val_loader = build_datasets()

    model = HarmonicNet(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Add computation graph on the first epoch ---
        if epoch == 1:
            sample_input = torch.zeros(1, 1)
            writer.add_graph(model, sample_input)

        # --- Training step ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_preds = model(x_val)
                val_loss = criterion(val_preds, y_val).item()

        # --- Log scalars ---
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)

        # --- Save best model based on validation loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch:>3d}  train={train_loss:.5f}  val={val_loss:.5f}  ** saved best **")
        elif epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3d}  train={train_loss:.5f}  val={val_loss:.5f}")

    writer.close()
    print(f"\nTraining finished.  Best val loss: {best_val_loss:.5f}")
    print(f"Best model saved to: {Path(best_model_path).resolve()}")
    print(f"TensorBoard logs in: {Path(log_dir).resolve()}")
    print(f"\nRun `tensorboard --logdir {log_dir}` to visualise results.")


if __name__ == "__main__":
    train()
