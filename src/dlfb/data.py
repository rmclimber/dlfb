"""Small NumPy batch iterator shared by the chapter data pipelines."""

from collections.abc import Iterator, Mapping

import numpy as np


class NumpyDataset:
  """Batch aligned arrays without depending on TensorFlow.

  Evaluation visits every example once, including a short final batch. Training
  shuffles each epoch and repeats before batching, so even batches crossing an
  epoch boundary remain full. Each iterator has its own reproducible RNG.
  """

  def __init__(
    self,
    data: Mapping[str, np.ndarray],
    batch_size: int | None = None,
    *,
    is_training: bool = False,
    seed: int = 42,
  ):
    self.data = {name: np.asarray(values) for name, values in data.items()}
    if not self.data or any(values.ndim == 0 for values in self.data.values()):
      raise ValueError("Dataset must contain arrays with a sample dimension.")
    lengths = {len(values) for values in self.data.values()}
    if len(lengths) != 1:
      raise ValueError("Dataset arrays must have the same number of samples.")
    self.num_samples = lengths.pop()
    if not self.num_samples:
      raise ValueError("Dataset must contain at least one sample.")
    if batch_size is not None and (
      isinstance(batch_size, bool)
      or not isinstance(batch_size, (int, np.integer))
      or batch_size <= 0
    ):
      raise ValueError("batch_size must be a positive integer.")
    self.batch_size = self.num_samples if batch_size is None else int(batch_size)
    self.is_training = is_training
    self.seed = seed

  def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
    return self.as_numpy_iterator()

  def batch(self, batch_size: int, drop_remainder: bool = False):
    """Rebatch the underlying examples; retain the book's batching calls."""
    result = NumpyDataset(
      self.data, batch_size, is_training=self.is_training, seed=self.seed
    )
    result.drop_remainder = drop_remainder
    return result

  def as_numpy_iterator(self) -> Iterator[dict[str, np.ndarray]]:
    """Return independent batches; retain the book's iterator interface."""
    if not self.is_training:
      stop = self.num_samples
      if getattr(self, "drop_remainder", False):
        stop -= stop % self.batch_size
      for start in range(0, stop, self.batch_size):
        yield {
          name: values[start : start + self.batch_size]
          for name, values in self.data.items()
        }
      return

    rng = np.random.default_rng(self.seed)
    indices = rng.permutation(self.num_samples)
    cursor = 0
    while True:
      parts, needed = [], self.batch_size
      while needed:
        count = min(needed, self.num_samples - cursor)
        parts.append(indices[cursor : cursor + count])
        cursor += count
        needed -= count
        if cursor == self.num_samples:
          indices = rng.permutation(self.num_samples)
          cursor = 0
      batch_indices = np.concatenate(parts)
      yield {name: values[batch_indices] for name, values in self.data.items()}
