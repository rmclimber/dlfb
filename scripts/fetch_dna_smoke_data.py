"""Download the six pinned book CSVs used by the DNA notebook smoke test."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--destination", type=Path, required=True)
  args = parser.parse_args()
  manifest = Path(__file__).resolve().parents[1] / "tests/data/dna-sources.json"
  sources = json.loads(manifest.read_text())
  args.destination.mkdir(parents=True, exist_ok=True)
  for source in sources:
    target = args.destination / source["filename"]
    if target.is_file():
      content = target.read_bytes()
    else:
      response = requests.get(source["url"], timeout=120)
      response.raise_for_status()
      content = response.content
    if hashlib.sha256(content).hexdigest() != source["sha256"]:
      raise ValueError(f"Book data checksum differs: {source['filename']}")
    target.write_bytes(content)
    print(target.name, flush=True)


if __name__ == "__main__":
  main()
