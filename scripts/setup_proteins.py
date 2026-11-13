"""Install the locked protein profile into the interpreter running this script."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if sys.version_info[:2] not in {(3, 12), (3, 13)}:
    parser.error("The tested protein profiles require Python 3.12 or 3.13.")
  if args.backend != "cpu" and sys.platform != "linux":
    parser.error("CUDA profiles require Linux; select cpu on Windows.")
  root = Path(__file__).resolve().parents[1]
  requirements = root / "requirements" / "locks" / f"proteins-{args.backend}.txt"
  pip = [sys.executable, "-m", "pip"]
  if not args.dry_run:
    # Different plugin names survive a JAX downgrade/upgrade. Remove stale
    # registrations before installing the selected, version-matched backend.
    subprocess.run(
      pip
      + [
        "uninstall",
        "-y",
        "jax-cuda12-plugin",
        "jax-cuda12-pjrt",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
      ],
      check=True,
    )
  if not args.dry_run:
    subprocess.run(pip + ["uninstall", "-y", "torchvision", "torchaudio"], check=True)
  command = pip + ["install", "--require-hashes", "-r", str(requirements)]
  command += [
    "--extra-index-url",
    "https://download.pytorch.org/whl/" + ("cpu" if args.backend == "cpu" else "cu128"),
  ]
  if args.dry_run:
    command += ["--dry-run"]
  subprocess.run(command, check=True)
  if not args.dry_run:
    subprocess.run(pip + ["install", "--no-deps", "-e", str(root)], check=True)
    configuration = {
      "python": sys.version.split()[0],
      "backend": args.backend,
      "source": str(root),
    }
    (root / ".proteins-environment.json").write_text(json.dumps(configuration))
    print(json.dumps(configuration, indent=2))
    print("Installation complete. Restart the notebook session before importing dlfb.")


if __name__ == "__main__":
  main()
