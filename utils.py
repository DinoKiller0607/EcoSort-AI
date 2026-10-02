"""
File containing various utility functions for PyTorch model training.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)


def plot_loss_curves(results, compare=None):
  """Plots loss/accuracy/precision/recall/F1 curves from engine.train() results.

  Args:
    results: dict returned by engine.train().
    compare: optional dict {"label": [val_f1 per epoch]} drawn on the F1 panel
      (e.g. a previous run) for side-by-side comparison.
  """
  import matplotlib.pyplot as plt

  epochs = range(1, len(results["train_loss"]) + 1)
  panels = [("loss", "Loss"), ("acc", "Accuracy"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1 (macro)")]

  plt.figure(figsize=(18, 9))
  for i, (key, title) in enumerate(panels, start=1):
    plt.subplot(2, 3, i)
    plt.plot(epochs, results[f"train_{key}"], label=f"train_{key}")
    plt.plot(epochs, results[f"val_{key}"], label=f"val_{key}")
    if key == "f1" and compare:
      for label, values in compare.items():
        plt.plot(range(1, len(values) + 1), values, linestyle="--", label=label)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.legend()
  plt.tight_layout()
  plt.show()
