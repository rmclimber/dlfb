"""Execute the teaching cells with real CSV samples and short training budgets.

Requires the dev dependency group. Setup/install cells are excluded because this
runner executes in an already installed test environment. The source notebook's
training budgets are never changed on disk.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--notebook", type=Path, required=True)
  parser.add_argument("--data-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  notebook = nbformat.read(args.notebook, as_version=4)
  with tempfile.TemporaryDirectory(prefix="dlfb-dna-smoke-") as temp:
    assets = Path(temp)
    target = assets / "dna/datasets"
    target.mkdir(parents=True)
    for factor in ("CTCF", "ZNF24"):
      for split in ("train", "valid", "test"):
        name = f"{factor}_{split}_sequences.csv"
        frame = pd.read_csv(args.data_dir / name)
        sample = frame.groupby("label", group_keys=False).head(16)
        assert len(sample) == 32 and sample["label"].nunique() == 2
        sample.to_csv(target / name, index=False)
    setup = nbformat.v4.new_code_cell(
      f"import os\nos.environ['ASSETS_DIR'] = {str(assets)!r}\n"
      "os.environ['MPLBACKEND'] = 'Agg'\n"
      "from dlfb.utils.display import display\n"
      "import matplotlib\nmatplotlib.use('Agg')\n"
    )
    notebook.cells = [setup] + notebook.cells[9:]
    for cell in notebook.cells:
      if cell.cell_type == "code":
        cell.source = cell.source.replace("range(500)", "range(3)")
        cell.source = cell.source.replace("num_steps = 1000", "num_steps = 3")
        if cell.source.startswith("transcription_factors = ["):
          cell.source = 'transcription_factors = ["CTCF", "ZNF24"]'
        cell.outputs = []
        cell.execution_count = None
    kernel_root = assets / "kernels"
    kernel = kernel_root / "dna-smoke"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
      json.dumps(
        {
          "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
          ],
          "display_name": "DNA smoke test",
          "language": "python",
        }
      )
    )
    manager = KernelManager(
      kernel_name="dna-smoke",
      kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )

    def report(cell, cell_index):
      if cell.cell_type == "code":
        print(f"Cell {cell_index + 1}: {cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(
      notebook,
      km=manager,
      timeout=600,
      kernel_name="dna-smoke",
      on_cell_start=report,
      resources={"metadata": {"path": str(assets)}},
    )
    try:
      client.execute(cleanup_kc=True)
    finally:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      nbformat.write(notebook, args.output)
    count = sum(c.cell_type == "code" for c in notebook.cells)
    print(f"Executed {count} code cells with CTCF/ZNF24 CSV samples.")


if __name__ == "__main__":
  main()
