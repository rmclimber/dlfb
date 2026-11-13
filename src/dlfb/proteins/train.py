import jax
import numpy as np
import optax
import pandas as pd
from flax.training.train_state import TrainState
from sklearn import metrics
from tqdm import tqdm

from dlfb.data import NumpyDataset


def train(
  state: TrainState,
  dataset_splits: dict[str, NumpyDataset],
  batch_size: int,
  num_steps: int = 300,
  eval_every: int = 30,
):
  """Train model using NumPy batches and track performance metrics."""
  if num_steps <= 0 or eval_every <= 0 or batch_size <= 0:
    raise ValueError("Training counts must be positive.")
  # Create containers to handle calculated during training and evaluation.
  train_metrics, valid_metrics = [], []

  # Create batched dataset to pluck batches from for each step.
  train_batches = (
    dataset_splits["train"].batch(batch_size, drop_remainder=True).as_numpy_iterator()
  )

  steps = tqdm(range(num_steps))  # Steps with progress bar.
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    # Get batch of training data, convert into a JAX array, and train.
    state, loss = train_step(state, next(train_batches))
    train_metrics.append({"step": step, "loss": loss.item()})

    if step % eval_every == 0 or step == num_steps - 1:
      valid_metrics.append(
        {"step": step, **evaluate(state, dataset_splits["valid"], batch_size)}
      )

  return state, {"train": train_metrics, "valid": valid_metrics}


@jax.jit
def train_step(state, batch):
  """Run a single training step and update model parameters."""

  def calculate_loss(params):
    """Compute sigmoid cross-entropy loss from logits."""
    logits = state.apply_fn({"params": params}, x=batch["embedding"])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    return loss

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss


def eval_step(state, batch) -> dict[str, float]:
  """Run evaluation step and return mean metrics over targets."""
  logits = state.apply_fn({"params": state.params}, x=batch["embedding"])
  loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
  target_metrics = calculate_per_target_metrics(logits, batch["target"])
  metrics = {
    "loss": loss.item(),
    **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),
  }
  return metrics


def calculate_per_target_metrics(logits, targets):
  """Compute metrics for each target in a multi-label batch."""
  probs = jax.nn.sigmoid(logits)
  target_metrics = []
  for target, prob in zip(np.asarray(targets).T, np.asarray(probs).T):
    target_metrics.append(compute_metrics(target, prob))
  return target_metrics


def compute_metrics(
  targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> dict[str, float]:
  """Compute accuracy, recall, precision, auPRC, and auROC."""
  targets, probs = np.asarray(targets).ravel(), np.asarray(probs).ravel()
  return {
    "accuracy": float(metrics.accuracy_score(targets, probs >= thresh)),
    "recall": float(metrics.recall_score(targets, probs >= thresh, zero_division=0)),
    "precision": float(
      metrics.precision_score(targets, probs >= thresh, zero_division=0)
    ),
    "auprc": float(metrics.average_precision_score(targets, probs))
    if targets.sum()
    else 0.0,
    "auroc": float(metrics.roc_auc_score(targets, probs))
    if np.unique(targets).size == 2
    else float("nan"),
  }


def evaluate(state, dataset: NumpyDataset, batch_size: int = 32):
  """Compute each function's metrics over the entire finite split."""
  if dataset.is_training:
    raise ValueError("Evaluation requires a finite dataset.")
  targets, logits = [], []
  for batch in dataset.batch(batch_size):
    targets.append(batch["target"])
    logits.append(
      np.asarray(state.apply_fn({"params": state.params}, x=batch["embedding"]))
    )
  target, prediction = np.concatenate(targets), np.concatenate(logits)
  return {
    "loss": float(optax.sigmoid_binary_cross_entropy(prediction, target).mean()),
    **pd.DataFrame(calculate_per_target_metrics(prediction, target)).mean().to_dict(),
  }
