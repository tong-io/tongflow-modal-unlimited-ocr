"""Modal download entry for unlimited-ocr.

Run:
  modal run download.py::download

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import os

import modal

REPO_ID = "baidu/Unlimited-OCR"
MODEL_DIR = f"/models/{REPO_ID}"

volume = modal.Volume.from_name("models", create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub>=1.5.0,<2.0"
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping")
        return

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
