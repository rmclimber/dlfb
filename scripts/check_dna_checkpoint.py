"""Check supplied DNA checkpoint restoration and inference on the modern stack."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import optax

from dlfb.dna.dataset import load_dataset
from dlfb.dna.model import ConvModel, ConvModelV2
from dlfb.dna.train import train_step
from dlfb.utils.restore import restore, restore_for_inference


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--assets", type=Path, required=True)
  args = parser.parse_args()
  batch = load_dataset(args.assets / "dna/datasets/CTCF_valid_sequences.csv")
  x = batch["sequences"][:8]
  results = []
  for name, model in [("prototype", ConvModel()), ("CTCF", ConvModelV2())]:
    tx = optax.adam(optax.cosine_decay_schedule(0.001, 1000))
    initial = model.create_train_state(jax.random.PRNGKey(42), x, tx)
    loader = restore_for_inference if name == "prototype" else restore
    state, metrics = loader(args.assets / f"dna/models/{name}", initial)
    variables = {"params": state.params}
    kwargs = {}
    if name != "prototype":
      variables["batch_stats"] = state.batch_stats
      kwargs["is_training"] = False
    logits = np.asarray(state.apply_fn(variables, x, **kwargs))
    assert logits.shape == (8, 1) and np.isfinite(logits).all()
    resumed_step = None
    if name != "prototype":
      updated, training_metrics = train_step(
        state, {"sequences": x, "labels": batch["labels"][:8]}, jax.random.PRNGKey(7)
      )
      resumed_step = int(updated.step)
      assert resumed_step == int(state.step) + 1
      assert np.isfinite(float(training_metrics["loss"]))
    results.append(
      {
        "checkpoint": name,
        "mode": "weights-only" if name == "prototype" else "full-state",
        "optimizer_step": int(state.step),
        "resumed_step": resumed_step,
        "logits": logits.ravel().tolist(),
        "metrics_loaded": bool(metrics),
      }
    )
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
