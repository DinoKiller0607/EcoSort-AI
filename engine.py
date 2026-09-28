"""
Contains functions for training and validating a PyTorch model.
"""
from typing import Dict, List, Tuple
import torch
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float, float, float]:
    """Trains a PyTorch model for a single epoch."""
    model.train()

    train_loss = 0
    all_preds = []
    all_labels = []

    # Executes training loop; performs forward pass, backpropagation, and metric collection
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Accumulate predictions and labels for metric calculation
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        all_preds.extend(y_pred_class.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Adjust metrics
    train_loss = train_loss / len(dataloader)
    train_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    # Calculate Precision, Recall, and F1 Score (Macro Average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return train_loss, train_acc, precision, recall, f1


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float, float, float, float]:
    """Validates a PyTorch model for a single epoch."""
    model.eval()

    val_loss = 0
    all_preds = []
    all_labels = []

        # Iterates batches; computes predictions; accumulates loss and labels
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Accumulate predictions and labels
            val_pred_labels = val_pred_logits.argmax(dim=1)
            all_preds.extend(val_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Adjust metrics
    val_loss = val_loss / len(dataloader)
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    # Calculate Precision, Recall, and F1 Score (Macro Average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    return val_loss, val_acc, precision, recall, f1


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Trains and validates a PyTorch model."""
    results = {
        "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)
        results["train_f1"].append(train_f1)

        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_precision"].append(val_prec)
        results["val_recall"].append(val_rec)
        results["val_f1"].append(val_f1)

    return results