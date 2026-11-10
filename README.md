# dlfb

This repo contains the **`dlfb` companion library**, with utilities to simplify preprocessing, model building, training, and evaluation.

The library is installed as one of the setup steps in the Colab notebooks available in https://github.com/deep-learning-for-biology/notebooks.

## Foundation and DNA modernization (0.2)

The shared foundation and DNA chapter now support Python **3.12 and 3.13**.
Dependencies live in `pyproject.toml` and the tested versions are recorded in
`uv.lock`. The core is JAX/Flax/Optax/Orbax; DNA needs neither TensorFlow nor
PyTorch. The other four chapters have **not** been migrated or validated against
0.2; their legacy requirements files are retained for the later chapter passes.

```sh
uv sync --extra dna --locked --python 3.13
uv run --extra dna --locked pytest -q
uv pip check
```

For a runtime without development tools, add `--no-dev`. To use local Jupyter,
add `--extra notebook`. On Linux with a compatible NVIDIA GPU, choose exactly one
of `--extra cuda12` or `--extra cuda13`. CUDA dependency profiles are locked but
have not yet been execution-tested on a hosted Colab GPU. TPU is not a supported
DNA profile in this migration.

For pip-based Colab setup, use `python scripts/setup_dna.py --backend cpu`
(or `cuda12`/`cuda13`). It installs hash-pinned exports from `requirements/locks`,
selects a matching JAX plugin family, and stops on errors. Restart the session
after installation. The lock exports deliberately omit IPython/kernel packages;
Colab manages its own notebook shell. This is a chapter environment, not a promise
that every unrelated preinstalled Colab package remains compatible after upgrades.

The accompanying DNA notebook accepts the matching `dlfb-dna-modernized.zip`
source snapshot while the changes are unpublished. It does not fetch upstream
`main`. Do not use an older published copy of the library with that notebook.

### Alignment with the book

- Architectures, parameter names, loss functions, optimizers, and the original
  BatchNorm/GELU order are unchanged. Flax Linen is retained.
- `make_dataset` batches NumPy arrays. `convert_to_tfds` remains a compatibility
  alias for the book's calls, and `as_numpy_iterator()` remains available; this
  does not emulate TensorFlow's full Dataset API.
- Training shuffles reproducibly and repeats before batching. Validation/test
  include the final partial batch. Validation auROC is calculated over the full
  split, and the last training interval is recorded.
- New DNA checkpoints use `dna/models-modern/` in the notebook. `restore()`
  requires an identical training-state structure and keeps optimizer state.
  `restore_for_inference()` explicitly loads model parameters and BatchNorm state
  while preserving a fresh optimizer and step. Use it for legacy checkpoint
  optimizer layouts; it is not a training-resume operation.
- Numerical results can differ because of new library versions and seeded
  shuffling. Short smoke runs verify execution, not the book's final auROC.

### Repeatable notebook execution check

The runner executes every teaching code cell with 32 real examples per split,
CTCF and ZNF24, all six architecture variants, and three-step training budgets.
Setup/install cells are excluded. Original notebook budgets are unchanged.

```sh
uv run --extra dna --locked python scripts/fetch_dna_smoke_data.py --destination work/dna-data
uv run --extra dna --locked python scripts/smoke_dna_notebook.py --notebook ../dlfb_notebooks/notebooks/chapter_3_dna.ipynb --data-dir work/dna-data --output work/dna-smoke.ipynb
```

The downloaded CSVs are checked against `tests/data/dna-sources.json`. For the
supplied legacy checkpoint examples, download the prototype and CTCF model assets
and run `scripts/check_dna_checkpoint.py --assets PATH_TO_ASSETS`. Checkpoint
compatibility is separate from the reduced-data notebook execution check.

## Contributing

We welcome contributions!

- **Issues:** Use Issues only for reproducible bugs in the codebase (e.g. broken code, dependency errors, incorrect results). This separation helps keep bug reports from getting lost in general conversation.
- **Pull Requests:** Contributions that add new utilities, improve existing implementations, or expand the notebooks are very welcome! Please keep PRs focused and include a clear description of the change.

## Citations

If you use `dlfb` in your work and would like to cite it, you can cite the book like this:

```bibtex
@book{deep_learning_for_biology,
  title     = {Deep Learning for Biology},
  author    = {Ravarani, C. and Latysheva, N.},
  publisher = {O’Reilly Media},
  year      = {2025},
}
```

Or more plainly as "Ravarani & Latysheva, Deep Learning for Biology, O’Reilly Media, 2025."

## License

The **code** in this repository is licensed under the Apache 2.0 License.
