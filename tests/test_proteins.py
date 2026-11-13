import os
from pathlib import Path

import jax
import numpy as np
import optax
import pandas as pd
import pytest
import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmModel, EsmTokenizer

from dlfb.proteins.dataset import (
  build_dataset,
  get_mean_embeddings,
  make_dataset,
  store_sequence_embeddings,
)
from dlfb.proteins.inspect import MaskPredictor
from dlfb.proteins.model import Model
from dlfb.proteins.train import (
  calculate_per_target_metrics,
  compute_metrics,
  evaluate,
  train,
)
from dlfb.utils.restore import restore, store


@pytest.fixture
def esm(tmp_path):
  vocab = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "<null_1>",
    "<mask>",
  ]
  path = tmp_path / "vocab.txt"
  path.write_text("\n".join(vocab))
  tokenizer = EsmTokenizer(vocab_file=str(path))
  config = EsmConfig(
    vocab_size=len(vocab),
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    pad_token_id=1,
    mask_token_id=32,
  )
  torch.manual_seed(42)
  return tokenizer, EsmModel(config), EsmForMaskedLM(config)


def test_embedding_formula_and_masked_prediction(esm):
  tokenizer, model, masked = esm
  sequences = ["MALWM", "ACGT"]
  result = get_mean_embeddings(sequences, tokenizer, model, torch.device("cpu"))
  with torch.inference_mode():
    expected = (
      model(**tokenizer(sequences, padding=True, return_tensors="pt"))
      .last_hidden_state.mean(1)
      .numpy()
    )
  np.testing.assert_allclose(result, expected, atol=1e-6)
  predictor = MaskPredictor(tokenizer, masked)
  probs = np.asarray(predictor.predict("MALWM", 2))
  assert probs.shape == (33,) and np.isfinite(probs).all()
  assert probs.sum() == pytest.approx(1.0, abs=1e-6)
  with pytest.raises(ValueError):
    predictor.mask_sequence("ABC", 3)


def test_feather_batches_training_and_restore(esm, tmp_path):
  tokenizer, model, _ = esm
  frame = pd.DataFrame(
    {
      "EntryID": ["p" + str(i) for i in range(8)],
      "Sequence": ["MALWM", "ACGT"] * 4,
      "GO:1": [0, 1] * 4,
      "GO:2": [1, 0] * 4,
    },
    index=np.arange(8) + 100,
  )
  model.name_or_path = "test/esm"
  for split in ["train", "valid", "test"]:
    store_sequence_embeddings(
      frame, str(tmp_path / f"protein_dataset_{split}"), tokenizer, model, batch_size=3
    )
  splits = build_dataset(str(tmp_path / "protein_dataset"), "test/esm")
  mlp = Model(num_targets=2, dim=8)
  initial = mlp.create_train_state(
    jax.random.PRNGKey(42), next(iter(splits["train"]))["embedding"], optax.adam(0.01)
  )
  state, metrics = train(initial, splits, 3, num_steps=30, eval_every=7)
  assert int(state.step) == 30 and metrics["valid"][-1]["step"] == 29
  assert metrics["train"][-1]["loss"] < metrics["train"][0]["loss"]
  for size in [1, 3, 8]:
    actual = evaluate(state, splits["valid"], size)
    expected = evaluate(state, splits["valid"], 8)
    for key in actual:
      assert actual[key] == pytest.approx(expected[key], abs=1e-6)
  store(tmp_path / "checkpoint", state, metrics)
  recovered, saved = restore(tmp_path / "checkpoint", initial)
  assert saved == metrics and int(recovered.step) == 30


def test_target_axis_and_degenerate_metrics():
  y = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)
  logits = np.array([[-3, 3], [3, -3], [-3, 3]], dtype=np.float32)
  result = calculate_per_target_metrics(logits, y)
  assert len(result) == 2 and all(m["auroc"] == 1 for m in result)
  assert compute_metrics(np.zeros(3), np.zeros(3))["accuracy"] == 1
  assert np.isnan(compute_metrics(np.ones(3), np.ones(3))["auroc"])


def test_rebatching_keeps_tail_and_alignment():
  frame = pd.DataFrame({"ME:1": np.arange(5), "GO:1": [0, 1, 0, 1, 0]})
  ds = make_dataset(frame)
  assert [len(b["target"]) for b in ds.batch(2)] == [2, 2, 1]
  assert [len(b["target"]) for b in ds.batch(2, drop_remainder=True)] == [2, 2]
  assert next(iter(ds.batch(2)))["embedding"].dtype == np.float32


@pytest.mark.skipif(
  not os.environ.get("DLFB_ESM_TEST_PATH"),
  reason="Set path to pinned public ESM-2 8M snapshot",
)
def test_real_pretrained_esm():
  path = Path(os.environ["DLFB_ESM_TEST_PATH"])
  tokenizer = EsmTokenizer.from_pretrained(path, local_files_only=True)
  model = EsmModel.from_pretrained(path, local_files_only=True)
  result = get_mean_embeddings(["MALWM", "ACGT"], tokenizer, model)
  assert result.shape == (2, 320) and np.isfinite(result).all()
  masked = EsmForMaskedLM.from_pretrained(path, local_files_only=True)
  assert np.isfinite(MaskPredictor(tokenizer, masked).predict("MALWM", 2)).all()
