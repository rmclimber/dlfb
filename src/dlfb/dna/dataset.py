import numpy as np
import pandas as pd

from dlfb.data import NumpyDataset
from dlfb.dna.utils import dna_to_one_hot


def load_dataset_splits(
  path, transcription_factor, batch_size: int | None = None, *, seed: int = 42
):
  """Load train/valid/test splits using the book's CSVs and batch structure."""
  dataset_splits = {}
  for split in ["train", "valid", "test"]:
    dataset = load_dataset(
      sequence_db=f"{path}/{transcription_factor}_{split}_sequences.csv"
    )
    ds = make_dataset(dataset, batch_size, is_training=(split == "train"), seed=seed)
    dataset_splits.update({split: ds})
  return dataset_splits


def make_dataset(
  dataset, batch_size: int | None = None, is_training: bool = False, *, seed: int = 42
) -> NumpyDataset:
  """Batch DNA arrays, shuffling and repeating only the training split."""
  return NumpyDataset(dataset, batch_size, is_training=is_training, seed=seed)


def convert_to_tfds(
  dataset, batch_size: int | None = None, is_training: bool = False, *, seed: int = 42
) -> NumpyDataset:
  """Compatibility name from the book; now returns a NumPy batch iterator.

  The supported interface is as_numpy_iterator(), not the TensorFlow Dataset API.
  New code should use make_dataset().
  """
  return make_dataset(dataset, batch_size, is_training, seed=seed)


def load_dataset(sequence_db) -> dict[str, np.ndarray]:
  """Load sequences and labels from a CSV into numpy arrays."""
  df = pd.read_csv(sequence_db)
  return {
    "labels": df["label"].to_numpy(dtype=np.float32)[:, None],
    "sequences": np.array(
      [dna_to_one_hot(seq) for seq in df["sequence"]], dtype=np.float32
    ),
  }
