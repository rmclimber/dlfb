from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dlfb import provision


def test_modern_cli_help_and_chapter_option(monkeypatch):
  calls = []
  monkeypatch.setattr(provision, "provision_assets", lambda *args: calls.append(args))
  runner = CliRunner()
  assert runner.invoke(provision.app, ["--help"]).exit_code == 0
  result = runner.invoke(provision.app, ["--chapter", "dna", "--no-models"])
  assert result.exit_code == 0, result.output
  assert calls[0][0] == "dna" and calls[0][3] is False


def test_failed_download_stops_setup(monkeypatch, tmp_path):
  class FailedDownloader:
    def __init__(self, **kwargs):
      self.queued_downloads = False

    def enqueue_file(self, **kwargs):
      self.queued_downloads = True

    def download(self):
      return SimpleNamespace(errors=["network unavailable"])

  monkeypatch.setattr(provision, "Downloader", FailedDownloader)
  monkeypatch.setattr(
    provision, "get_sized_prefixes", lambda *args: [(1, "dna/datasets/test.csv")]
  )
  with pytest.raises(RuntimeError, match="Failed to download"):
    provision.provision_assets("dna", "https://example.com/", str(tmp_path), False, 100)
