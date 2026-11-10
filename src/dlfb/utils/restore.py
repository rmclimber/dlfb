from functools import wraps
from pathlib import Path

import jax
import orbax.checkpoint as ocp

from dlfb.log import log


def restore(store_path, state):
  with ocp.CheckpointManager(Path(store_path).resolve()) as mngr:
    state, metrics = restore_checkpoint(mngr, state)
  return state, metrics


def store(store_path, state, metrics) -> None:
  with ocp.CheckpointManager(Path(store_path).resolve()) as mngr:
    save_final_checkpoint(mngr, state, metrics)


def restore_for_inference(store_path, state):
  """Load model weights without restoring the optimizer or training step.

  Use this explicitly for historical checkpoints whose optimizer serialization
  differs from the installed Optax/Orbax versions. Full restore() remains strict
  so a training-state mismatch cannot silently reset or discard optimizer data.
  """
  model_state = {"params": state.params}
  if hasattr(state, "batch_stats"):
    model_state["batch_stats"] = state.batch_stats
  abstract = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, model_state)
  step_path = Path(store_path).resolve() / "0"
  with ocp.PyTreeCheckpointer() as checkpointer:
    restored = checkpointer.restore(
      step_path / "state",
      args=ocp.args.PyTreeRestore(
        item=abstract,
        restore_args=ocp.checkpoint_utils.construct_restore_args(model_state),
        partial_restore=True,
      ),
    )
  with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
    metrics = checkpointer.restore(step_path / "extra_metadata")
  return state.replace(**restored), metrics


def restorable(train_fn):
  # NOTE: wraps required for proper display with 'inspect'
  @wraps(train_fn)
  def wrapper(state, store_path: str | None = None, **kwargs):
    if store_path:
      checkpoint_path = Path(store_path).resolve()
      with ocp.CheckpointManager(checkpoint_path) as mngr:
        if (checkpoint_path / "0").exists():
          state, metrics = restore_checkpoint(mngr, state)
        else:
          log.debug("Training new model checkpointing state and metrics...")
          state, metrics = train_fn(state, **kwargs)
          save_final_checkpoint(mngr, state, metrics)
    else:
      log.debug("Training new model without checkpointing...")
      state, metrics = train_fn(state, **kwargs)
    return state, metrics

  return wrapper


def restore_checkpoint(mngr: ocp.CheckpointManager, state):
  # NOTE: benefit of approach is that we can reconstruct a model
  #       built on a GPU back in a CPU environment.
  #       see: https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpointing_pytrees.html#change-sharding
  restored = mngr.restore(
    0,
    args=ocp.args.Composite(
      state=ocp.args.StandardRestore(get_abstract_state(state)),
      extra_metadata=ocp.args.JsonRestore(),
    ),
  )
  log.debug("Train state and metrics restored from checkpoint")
  state, metrics = restored.state, restored.extra_metadata
  return state, metrics


def get_abstract_state(state):
  """Returns a shape/dtype-only version of the model's state"""
  return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)


def save_final_checkpoint(mngr, state, metrics):
  mngr.save(
    0,
    args=ocp.args.Composite(
      state=ocp.args.StandardSave(state),
      extra_metadata=ocp.args.JsonSave(metrics),
    ),
  )
  mngr.wait_until_finished()
