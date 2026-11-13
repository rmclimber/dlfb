# Protein chapter migration

The protein chapter keeps the book's ESM-2 checkpoints (650M token visualization,
150M masked prediction and function embeddings, and 8M localization embeddings),
Flax Linen MLP, sigmoid cross-entropy, Adam, and section order. The MLP architecture
and parameter names are unchanged. Mean embeddings retain the book's literal
mean over token positions, including special/padded tokens; this migration does
not silently change the representation or make it independent of batch padding.

## Environments

Python 3.12 and 3.13 are supported by the lock. Use exactly one protein extra:

```sh
# CPU PyTorch and CPU JAX, with the test tools.
uv sync --extra proteins --extra dna --locked --python 3.13
uv run --extra proteins --extra dna --locked pytest -q

# Linux GPU PyTorch (CUDA 12.8) and JAX CUDA 12.
uv sync --extra proteins-gpu --extra cuda12 --locked --python 3.13
```

PyTorch 2.11, Transformers 4.57.6, PyArrow, BioPython, obonet, and the chapter's
plotting tools replace the old protein requirements. Torch uses explicit CPU or
CUDA indexes in uv. TensorFlow, tensorflow-datasets, Hugging Face datasets, and
torchvision are absent from the chapter profile. Plain pip does not interpret uv
index configuration: use the profile installer for the exported pip locks.

For Colab, run `scripts/setup_proteins.py --backend gpu` or `--backend cpu` using
the notebook kernel's Python, then restart. The installer removes stale JAX
plugins and mismatched optional torchvision/torchaudio installations. The notebook
checks both Torch CUDA and JAX devices. JAX preallocation is disabled so the two
frameworks can share GPU memory. GPU execution requires a separate hosted check.

## Book compatibility and corrected evaluation

`make_dataset` replaces TensorFlow transport with the shared NumPy iterator. The
old `convert_to_tfds` name is an alias. Protein iterators return batches of one by
default; `.batch(n)` selects a batch size, and `.as_numpy_iterator()` remains
available. Training shuffles full epochs reproducibly instead of using the old
50-example shuffle buffer. Test/validation retain their final partial batch.

The old per-target metric helper iterated over rows. It now evaluates GO-function
columns. Validation scores cover the whole split rather than averaging batch
auROCs. All-negative accuracy is computed normally, average precision is zero
when there are no positives, and auROC is NaN for a single-class target. These
corrections can change reported metrics. The final training interval is logged.

`MaskPredictor` uses evaluation/inference mode and moves its inputs and outputs
across the selected Torch device safely. Feather writing handles non-default
DataFrame indexes. New MLP checkpoints use `proteins/models-modern/`.

## Notebook execution check

```sh
DLFB_ESM_TEST_PATH=/path/to/esm8m uv run --extra proteins --extra dna --locked pytest -q
uv run --extra proteins --extra dna --locked python scripts/smoke_proteins_notebook.py --notebook ../dlfb_notebooks/notebooks/chapter_2_proteins.ipynb --assets /path/to/protein-assets --output proteins-smoke.ipynb
```

The real-model fixture is `facebook/esm2_t6_8M_UR50D` revision
`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`. Assets include that snapshot under
`esm8m/` and the book's localization CSV, GO descriptions, and original training
Feather file under `proteins/datasets/`. The smoke runner reconstructs FASTA,
taxonomy and GO annotation fixtures from 120 actual book training examples,
uses two real localization groups, substitutes 8M at larger model load sites,
lowers the fixture's function-frequency threshold, and takes three training steps.
This exercises the teaching code without claiming a full 650M/150M validation
or reproducing the book's final model quality. Setup/install cells are tested
separately from teaching code. No other chapter migration is included.
