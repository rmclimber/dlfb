from typing import Any

import flax.linen as nn
import jax
import numpy as np
import optax
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from dlfb.data import NumpyDataset
from dlfb.dna.model import TrainStateWithBatchNorm
from dlfb.utils.metrics_logger import MetricsLogger
from dlfb.utils.restore import restorable


@restorable
def train(
  state: TrainStateWithBatchNorm,
  rng: jax.Array,
  dataset_splits: dict[str, NumpyDataset],
  num_steps: int,
  eval_every: int = 100,
) -> tuple[TrainStateWithBatchNorm, Any]:
  """Train a model and log metrics over steps."""
  if num_steps <= 0 or eval_every <= 0:
    raise ValueError("num_steps and eval_every must be positive.")
  metrics = MetricsLogger()
  train_batches = dataset_splits["train"].as_numpy_iterator()

  steps = tqdm(range(num_steps))  # Steps with progress bar.
  for step in steps:
    steps.set_description(f"Step {step + 1}")

    rng, rng_dropout = jax.random.split(rng, 2)
    train_batch = next(train_batches)
    state, batch_metrics = train_step(state, train_batch, rng_dropout)
    metrics.log_step(split="train", **batch_metrics)

    if step % eval_every == 0 or step == num_steps - 1:
      # auROC must be computed across the split, not averaged over batches.
      batch_metrics = evaluate(state, dataset_splits["valid"])
      metrics.log_step(split="valid", **batch_metrics)
      metrics.flush(step=step)

    steps.set_postfix_str(metrics.latest(["loss"]))

  return state, metrics.export()


@jax.jit
def train_step(state, batch, rng_dropout: jax.Array):
  """Run a training step and update parameters."""

  def calculate_loss(params, batch):
    """Make predictions on batch and compute binary cross-entropy loss."""
    logits, updates = state.apply_fn(
      {"params": params, "batch_stats": state.batch_stats},
      x=batch["sequences"],
      is_training=True,
      rngs={"dropout": rng_dropout},
      mutable=["batch_stats"],
    )

    loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

    return loss, updates

  grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
  (loss, updates), grads = grad_fn(state.params, batch)
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=updates["batch_stats"])

  metrics = {"loss": loss}

  return state, metrics


def eval_step(state, batch):
  """Evaluate model on a single batch."""
  logits = state.apply_fn(
    {"params": state.params, "batch_stats": state.batch_stats},
    x=batch["sequences"],
    is_training=False,
    mutable=False,
  )
  loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
  metrics = {
    "loss": loss.item(),
    **compute_metrics(batch["labels"], logits),
  }
  return metrics


def compute_metrics(y_true: np.ndarray, logits: np.ndarray):
  """Compute accuracy and auROC for model predictions."""
  y_true = np.asarray(y_true).reshape(-1)
  logits = np.asarray(logits).reshape(-1)
  metrics = {
    "accuracy": accuracy_score(y_true, nn.sigmoid(logits) >= 0.5),
    # auROC is undefined for a split containing only one class.
    "auc": float(roc_auc_score(y_true, logits))
    if np.unique(y_true).size == 2
    else float("nan"),
  }
  return metrics


def evaluate(state, dataset: NumpyDataset):
  """Evaluate every sample, including the final partial batch."""
  if dataset.is_training:
    raise ValueError("Evaluation requires a finite, non-training dataset.")
  labels, predictions = [], []
  for batch in dataset.as_numpy_iterator():
    logits = state.apply_fn(
      {"params": state.params, "batch_stats": state.batch_stats},
      x=batch["sequences"],
      is_training=False,
      mutable=False,
    )
    labels.append(batch["labels"])
    predictions.append(np.asarray(logits))
  labels, logits = np.concatenate(labels), np.concatenate(predictions)
  return {
    "loss": float(optax.sigmoid_binary_cross_entropy(logits, labels).mean()),
    **compute_metrics(labels, logits),
  }
