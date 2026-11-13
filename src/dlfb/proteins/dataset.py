import os
from math import ceil

import numpy as np
import obonet
import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from dlfb.data import NumpyDataset
from dlfb.proteins.utils import get_device


def build_dataset(
  store_file_prefix: str, model_checkpoint: str
) -> dict[str, NumpyDataset]:
  """Build train/valid/test NumPy batch iterators from stored embeddings."""
  dataset_splits = {}

  for split in ["train", "valid", "test"]:
    dataset_splits[split] = make_dataset(
      df=load_sequence_embeddings(
        store_file_prefix=f"{store_file_prefix}_{split}",
        model_checkpoint=model_checkpoint,
      ),
      is_training=(split == "train"),
    )
  return dataset_splits


def make_dataset(
  df: pd.DataFrame,
  embeddings_prefix="ME:",
  target_prefix="GO:",
  is_training=False,
  shuffle_buffer=50,
  *,
  seed=42,
) -> NumpyDataset:
  """Make float32 batches; shuffle_buffer is retained for old book calls.

  NumPy shuffles complete epochs with a seed. It does not emulate TensorFlow's
  bounded shuffle buffer. Use .batch(n) to choose a batch size.
  """
  data = {
    "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(dtype=np.float32),
    "target": df.filter(regex=f"^{target_prefix}").to_numpy(dtype=np.float32),
  }
  if not data["embedding"].shape[1] or not data["target"].shape[1]:
    raise ValueError("Expected embedding and GO target columns.")
  return NumpyDataset(data, 1, is_training=is_training, seed=seed)


def convert_to_tfds(*args, **kwargs) -> NumpyDataset:
  """Book compatibility alias; the result supports batching and NumPy iteration."""
  return make_dataset(*args, **kwargs)


def load_sequence_embeddings(
  store_file_prefix: str, model_checkpoint: str
) -> pd.DataFrame:
  """Load stored embedding DataFrame from disk."""
  model_name = model_checkpoint.replace("/", "_")
  store_file = f"{store_file_prefix}_{model_name}.feather"
  return pd.read_feather(store_file)


def store_sequence_embeddings(
  sequence_df: pd.DataFrame,
  store_prefix: str,
  tokenizer: PreTrainedTokenizer,
  model: PreTrainedModel,
  batch_size: int = 64,
  force: bool = False,
) -> None:
  """Extract and store mean embeddings for each protein sequence."""
  if batch_size <= 0 or sequence_df.empty:
    raise ValueError("Embedding extraction requires data and a positive batch size.")
  model_name = str(model.name_or_path).replace("/", "_")
  store_file = f"{store_prefix}_{model_name}.feather"

  if not os.path.exists(store_file) or force:
    device = get_device()

    # Iterate through protein dataframe in batches, extracting embeddings.
    n_batches = ceil(sequence_df.shape[0] / batch_size)
    batches: list[np.ndarray] = []
    for i in range(n_batches):
      batch_seqs = list(
        sequence_df["Sequence"].iloc[i * batch_size : (i + 1) * batch_size]
      )
      batches.extend(get_mean_embeddings(batch_seqs, tokenizer, model, device))

    # Store each of the embedding values in a separate column in the dataframe.
    embeddings = pd.DataFrame(np.vstack(batches))
    embeddings.columns = [f"ME:{int(i) + 1}" for i in range(embeddings.shape[1])]
    df = pd.concat([sequence_df.reset_index(drop=True), embeddings], axis=1)
    df.to_feather(store_file)


def get_mean_embeddings(
  sequences: list[str],
  tokenizer: PreTrainedTokenizer,
  model: PreTrainedModel,
  device: torch.device | None = None,
) -> np.ndarray:
  """Compute mean embedding for each sequence using a protein LM."""
  if not sequences:
    raise ValueError("At least one protein sequence is required.")
  if device is None:
    device = get_device()

  # Tokenize input sequences and pad them to equal length.
  model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

  # Move tokenized inputs to the target device (CPU or GPU).
  model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

  # Move model to the target device and set it to evaluation mode.
  model = model.to(device)
  model.eval()

  # Forward pass without gradient tracking to obtain embeddings.
  with torch.inference_mode():
    outputs = model(**model_inputs)
    mean_embeddings = outputs.last_hidden_state.mean(dim=1)

  return mean_embeddings.detach().cpu().numpy()


def get_go_term_descriptions(store_path: str) -> pd.DataFrame:
  """Return GO term to description mapping, downloading if needed."""
  if not os.path.exists(store_path):
    url = "https://current.geneontology.org/ontology/go-basic.obo"
    graph = obonet.read_obo(url)

    # Extract GO term IDs and names from the graph nodes.
    id_to_name = {id: data.get("name") for id, data in graph.nodes(data=True)}
    go_term_descriptions = pd.DataFrame(
      id_to_name.items(),
      columns=["term", "description"],
    )
    go_term_descriptions.to_csv(store_path, index=False)

  else:
    go_term_descriptions = pd.read_csv(store_path)
  return go_term_descriptions
