import importlib.util

import jax
import numpy as np
import optax
import pytest

from dlfb.dna.dataset import make_dataset
from dlfb.dna.model import ConvModel, ConvModelV2, ConvTransformerModel
from dlfb.dna.train import compute_metrics, eval_step, evaluate, train, train_step
from dlfb.dna.utils import compute_input_gradient
from dlfb.utils.restore import restorable, restore, restore_for_inference, store


@pytest.fixture
def batch():
  rng = np.random.default_rng(42)
  x = np.eye(4, dtype=np.float32)[rng.integers(4, size=(8, 32))]
  return {"sequences": x, "labels": np.array([0, 1] * 4, np.float32)[:, None]}


def new_state(model, batch):
  return model.create_train_state(
    jax.random.PRNGKey(42), batch["sequences"], optax.adam(0.01)
  )


def test_dna_environment_does_not_require_tensorflow():
  assert importlib.util.find_spec("tensorflow") is None


def test_prototype_gradient_and_optimizer_match_book(batch):
  model = ConvModel(conv_filters=4, dense_units=8)
  state = new_state(model, batch)

  def loss(params):
    return optax.sigmoid_binary_cross_entropy(
      model.apply({"params": params}, batch["sequences"]), batch["labels"]
    ).mean()

  initial = float(loss(state.params))
  step = jax.jit(lambda s: s.apply_gradients(grads=jax.grad(loss)(s.params)))
  for _ in range(30):
    state = step(state)
  assert float(loss(state.params)) < initial * 0.8
  gradient = compute_input_gradient(state, batch["sequences"][0])
  assert gradient.shape == (32, 4)
  assert np.isfinite(gradient).all()


@pytest.mark.parametrize(
  "model",
  [
    ConvModelV2(conv_filters=8, dense_units=16),
    ConvTransformerModel(conv_filters=8, dense_units=16),
    ConvTransformerModel(conv_filters=8, dense_units=16, num_mlp_blocks=0),
    ConvTransformerModel(
      conv_filters=8, dense_units=16, num_conv_blocks=1, num_mlp_blocks=0
    ),
    ConvTransformerModel(
      conv_filters=8,
      dense_units=16,
      num_transformer_blocks=1,
      num_transformer_heads=2,
      transformer_dense_units=8,
    ),
    ConvTransformerModel(
      conv_filters=8,
      dense_units=16,
      num_transformer_blocks=2,
      num_transformer_heads=2,
      transformer_dense_units=8,
    ),
  ],
)
def test_model_forward_backward_and_batch_statistics(model, batch):
  state = new_state(model, batch)
  initial_stats = jax.tree_util.tree_leaves(state.batch_stats)
  updated, metrics = train_step(state, batch, jax.random.PRNGKey(7))
  assert int(updated.step) == 1
  assert np.isfinite(float(metrics["loss"]))
  assert any(
    not np.array_equal(a, b)
    for a, b in zip(initial_stats, jax.tree_util.tree_leaves(updated.batch_stats))
  )
  assert np.isfinite(list(eval_step(updated, batch).values())).all()


def test_evaluation_is_independent_of_batch_partition(batch):
  state = new_state(ConvModelV2(conv_filters=4, dense_units=8), batch)
  whole = evaluate(state, make_dataset(batch))
  chunks = evaluate(state, make_dataset(batch, 3))
  for metric in whole:
    assert chunks[metric] == pytest.approx(whole[metric], abs=1e-6)
  assert np.isnan(compute_metrics(np.zeros(2), np.ones(2))["auc"])
  with pytest.raises(ValueError, match="finite"):
    evaluate(state, make_dataset(batch, is_training=True))


def test_checkpoint_roundtrip_and_cached_training(tmp_path, batch):
  initial = new_state(ConvModelV2(conv_filters=4, dense_units=8), batch)
  splits = {"train": make_dataset(batch, 4, True), "valid": make_dataset(batch, 3)}
  kwargs = {
    "rng": jax.random.PRNGKey(4),
    "dataset_splits": splits,
    "num_steps": 3,
    "eval_every": 100,
    "store_path": str(tmp_path / "model"),
  }
  trained, metrics = train(state=initial, **kwargs)
  assert int(trained.step) == 3
  assert metrics["valid"]["auc"][-1]["round"] == 2
  recovered, recovered_metrics = restore(tmp_path / "model", initial)
  assert recovered_metrics == metrics
  for expected, actual in zip(
    jax.tree_util.tree_leaves(trained), jax.tree_util.tree_leaves(recovered)
  ):
    np.testing.assert_array_equal(actual, expected)
  # No further optimizer steps should run when the checkpoint exists.
  cached, _ = train(state=initial, **{**kwargs, "num_steps": 5})
  assert int(cached.step) == 3


def test_prototype_checkpoint_roundtrip(tmp_path, batch):
  initial = new_state(ConvModel(conv_filters=4, dense_units=8), batch)
  store(tmp_path / "prototype", initial, {"loss": 0.5})
  recovered, metrics = restore(tmp_path / "prototype", initial)
  assert metrics == {"loss": 0.5}
  np.testing.assert_array_equal(
    recovered.apply_fn({"params": recovered.params}, batch["sequences"]),
    initial.apply_fn({"params": initial.params}, batch["sequences"]),
  )


def test_corrupt_checkpoint_does_not_silently_retrain(tmp_path):
  (tmp_path / "0").mkdir()

  @restorable
  def unexpected_training(state):
    pytest.fail("Corrupt checkpoints must not be silently overwritten")

  with pytest.raises(Exception) as error:
    unexpected_training(state=None, store_path=str(tmp_path))
  assert not isinstance(error.value, AssertionError)


def test_inference_restore_explicitly_preserves_fresh_optimizer(tmp_path, batch):
  model = ConvModelV2(conv_filters=4, dense_units=8)
  trained, _ = train_step(new_state(model, batch), batch, jax.random.PRNGKey(3))
  store(tmp_path / "trained", trained, {"source": "test"})
  fresh = model.create_train_state(
    jax.random.PRNGKey(1), batch["sequences"], optax.sgd(0.1)
  )
  recovered, _ = restore_for_inference(tmp_path / "trained", fresh)
  assert int(recovered.step) == 0
  assert recovered.opt_state == fresh.opt_state
  for field in ("params", "batch_stats"):
    for expected, actual in zip(
      jax.tree_util.tree_leaves(getattr(trained, field)),
      jax.tree_util.tree_leaves(getattr(recovered, field)),
    ):
      np.testing.assert_array_equal(actual, expected)
