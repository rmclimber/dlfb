"""Execute all protein teaching cells with real book data and pinned ESM-2 8M.

The source notebook keeps its 650M/150M models and training budgets. This CPU
execution check substitutes 8M at load sites and uses 120 annotated proteins.
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
  parser.add_argument("--assets", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  source = args.assets.resolve()
  nb = nbformat.read(args.notebook, as_version=4)
  with tempfile.TemporaryDirectory(prefix="proteins-smoke-") as temp:
    assets = Path(temp)
    dest = assets / "proteins/datasets"
    dest.mkdir(parents=True)
    data = source / "proteins/datasets"
    frame = pd.read_feather(
      data / "protein_dataset_train_facebook_esm2_t30_150M_UR50D.feather"
    ).head(120)
    columns = [c for c in frame if c.startswith("GO:")]
    labels = frame[["EntryID"] + columns].melt(
      "EntryID", var_name="term", value_name="value"
    )
    labels = labels[labels.value == 1].assign(aspect="MFO")
    labels[["EntryID", "term", "aspect"]].to_csv(
      dest / "train_terms.tsv.zip", sep="\t", index=False
    )
    pd.DataFrame({"EntryID": frame.EntryID, "taxonomyID": 9606}).to_csv(
      dest / "train_taxonomy.tsv.zip", sep="\t", index=False
    )
    (dest / "train_sequences.fasta").write_text(
      "".join(f">{row.EntryID}\n{row.Sequence}\n" for row in frame.itertuples())
    )
    pd.read_csv(data / "go_term_descriptions.csv").to_csv(
      dest / "go_term_descriptions.csv", index=False
    )
    location = pd.read_csv(data / "sequence_df_cco.csv")
    location = location[~location.term.isin(["GO:0005575", "GO:0110165"])]
    single = location.groupby("EntryID").term.nunique()
    location = location[location.EntryID.isin(single[single == 1].index)]
    location = location[
      location.term.isin(["GO:0005576", "GO:0016020"])
      & location.Length.between(100, 500)
    ]
    location.groupby("term").head(20).to_csv(dest / "sequence_df_cco.csv", index=False)
    preamble = nbformat.v4.new_code_cell(
      f"import os\nos.environ['ASSETS_DIR']={str(assets)!r}\nos.environ['MPLBACKEND']='Agg'\n"
      "import torch\ntorch.set_num_threads(2)\nfrom dlfb.utils.display import display\n"
    )
    nb.cells = [preamble] + nb.cells[9:]
    model_path = str(source / "esm8m")
    for cell in nb.cells:
      if cell.cell_type != "code":
        continue
      cell.metadata["tags"] = [
        tag for tag in cell.metadata.get("tags", []) if tag != "skip-execution"
      ]
      for name in [
        "facebook/esm2_t33_650M_UR50D",
        "facebook/esm2_t30_150M_UR50D",
        "facebook/esm2_t6_8M_UR50D",
      ]:
        cell.source = cell.source.replace(name, model_path)
      cell.source = cell.source.replace("num_steps=300", "num_steps=3").replace(
        ">= 50", ">= 2"
      )
      cell.outputs = []
      cell.execution_count = None
    kernel_root = assets / "kernels"
    kernel = kernel_root / "proteins-smoke"
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
          "display_name": "Proteins smoke",
          "language": "python",
        }
      )
    )
    manager = KernelManager(
      kernel_name="proteins-smoke",
      kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )

    def report(cell, cell_index):
      if cell.cell_type == "code":
        print(f"Cell {cell_index + 1}: {cell.source.splitlines()[0]}", flush=True)

    client = NotebookClient(
      nb,
      km=manager,
      kernel_name="proteins-smoke",
      timeout=600,
      on_cell_start=report,
      resources={"metadata": {"path": str(assets)}},
    )
    try:
      client.execute(cleanup_kc=True)
    finally:
      nbformat.write(nb, args.output)
    print("Passed", sum(c.cell_type == "code" for c in nb.cells), "code cells.")


if __name__ == "__main__":
  main()
