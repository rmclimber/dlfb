from itertools import islice

import numpy as np
import pytest

from dlfb.data import NumpyDataset
from dlfb.dna.dataset import convert_to_tfds, load_dataset_splits, make_dataset


def examples(n=5):
  return {"sequences": np.arange(n)[:, None], "labels": np.arange(n)[:, None]}


def test_evaluation_retains_last_batch_and_is_repeatable():
  ds = NumpyDataset(examples(), 2)
  for _ in range(2):
    batches = list(ds)
    assert [len(b["labels"]) for b in batches] == [2, 2, 1]
    np.testing.assert_array_equal(
      np.concatenate([b["labels"] for b in batches]), examples()["labels"]
    )


@pytest.mark.parametrize("batch_size", [2, 5, 13])
def test_training_repeats_before_batching_without_losing_alignment(batch_size):
  ds = NumpyDataset(examples(), batch_size, is_training=True, seed=3)
  batches = list(islice(ds, 10))
  assert all(len(b["labels"]) == batch_size for b in batches)
  for batch in batches:
    np.testing.assert_array_equal(batch["sequences"], batch["labels"])
  stream = np.concatenate([b["labels"] for b in batches]).ravel()
  for start in range(0, len(stream) - 4, 5):
    assert sorted(stream[start : start + 5]) == list(range(5))
  np.testing.assert_array_equal(batches[0]["labels"], next(iter(ds))["labels"])


@pytest.mark.parametrize(
  "data,size",
  [
    ({}, 1),
    ({"a": np.array(1)}, 1),
    ({"a": np.array([])}, 1),
    ({"a": np.ones(2), "b": np.ones(3)}, 1),
    (examples(), 0),
    (examples(), -1),
    (examples(), 1.5),
    (examples(), True),
  ],
)
def test_invalid_data_fails_instead_of_hanging(data, size):
  with pytest.raises(ValueError):
    NumpyDataset(data, size, is_training=True)


def test_book_compatibility_alias():
  old = next(convert_to_tfds(examples(), 3, True, seed=7).as_numpy_iterator())
  new = next(make_dataset(examples(), 3, True, seed=7).as_numpy_iterator())
  np.testing.assert_array_equal(old["sequences"], new["sequences"])


def test_csv_splits_preserve_encoding_shapes_and_float32(tmp_path):
  for split in ("train", "valid", "test"):
    (tmp_path / f"CTCF_{split}_sequences.csv").write_text(
      "sequence,label\nACGTN,0\nTTTTA,1\n", encoding="utf-8"
    )
  splits = load_dataset_splits(tmp_path, "CTCF", 3)
  batch = next(iter(splits["valid"]))
  assert batch["sequences"].shape == (2, 5, 4)
  assert batch["labels"].shape == (2, 1)
  assert batch["sequences"].dtype == batch["labels"].dtype == np.float32
  np.testing.assert_array_equal(batch["sequences"][0, -1], [1, 1, 1, 1])
  assert next(iter(splits["train"]))["labels"].shape == (3, 1)
