"""Modal deploy entry for unlimited-ocr.

Document-to-text (OCR / long-horizon document parsing) with Baidu's
``baidu/Unlimited-OCR`` model, running on a GPU.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import modal
from tongflow import deploy

from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset_as_path
from tongflow.slots import node_slot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Plugin-internal config; not exposed via ABI.
REPO_ID = "baidu/Unlimited-OCR"
MODEL_DIR = f"/models/{REPO_ID}"
PDF_DPI = 300
MAX_LENGTH = 32768
NO_REPEAT_NGRAM_SIZE = 35
# gundam config (single image): high-detail crop tiling.
SINGLE_BASE_SIZE = 1024
SINGLE_IMAGE_SIZE = 640
SINGLE_NGRAM_WINDOW = 128
# base config (multi-page / PDF): no cropping, larger ngram window.
MULTI_IMAGE_SIZE = 1024
MULTI_NGRAM_WINDOW = 1024
SINGLE_PROMPT = "<image>document parsing."
MULTI_PROMPT = "<image>Multi page parsing."

volume = modal.Volume.from_name("models", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .pip_install(
        "tongflow==0.1.0",
        # transformers pulls a compatible huggingface_hub (<1.0) and tokenizers;
        # do not pin those here or they conflict with transformers' own ranges.
        "transformers==4.57.1",
        "einops==0.8.2",
        "addict==2.4.0",
        "easydict==1.13",
        "pymupdf==1.27.2.2",
        "matplotlib==3.10.8",
        "psutil==7.2.2",
        "pillow",
        "numpy",
        "tqdm",
    )
)

app = modal.App(Path(__file__).resolve().parent.name, image=image)
secrets = modal.Secret.from_dict({})

with image.imports():
    import torch
    from transformers import AutoModel, AutoTokenizer


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = PDF_DPI) -> list[str]:
    """Render each PDF page to a PNG; return the page image paths in order."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths: list[str] = []
    for i, page in enumerate(doc):
        out = str(Path(out_dir) / f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def is_pdf(path: str) -> bool:
    """Detect a PDF by its magic bytes (the input Asset has no reliable suffix)."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


@deploy
@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    timeout=1800,
    secrets=[secrets],
    scaledown_window=30,
)
class Inference:
    @modal.enter()
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        # MHA path only registers "mha_eager" (sliding-window attention); flash /
        # sdpa are not wired for this model, so eager is required.
        self.model = (
            AutoModel.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .eval()
            .cuda()
        )

    def _parse_single(self, image_path: str, out_dir: str) -> str:
        return self.model.infer(
            self.tokenizer,
            prompt=SINGLE_PROMPT,
            image_file=image_path,
            output_path=out_dir,
            base_size=SINGLE_BASE_SIZE,
            image_size=SINGLE_IMAGE_SIZE,
            crop_mode=True,
            max_length=MAX_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            ngram_window=SINGLE_NGRAM_WINDOW,
            eval_mode=True,
        )

    def _parse_multi(self, image_paths: list[str], out_dir: str) -> str:
        outputs, _tokens = self.model.infer_multi(
            self.tokenizer,
            prompt=MULTI_PROMPT,
            image_files=image_paths,
            output_path=out_dir,
            image_size=MULTI_IMAGE_SIZE,
            max_length=MAX_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            ngram_window=MULTI_NGRAM_WINDOW,
            save_results=False,
        )
        return outputs

    @modal.method()
    @node_slot(NodeSlots.PARSE_DOCUMENT)
    def parse_document(self, input: ParseDocumentInput) -> ParseDocumentOutput:
        if input.document is None:
            return ParseDocumentOutput(success=False, error="Missing `document` Asset")
        try:
            with asset_as_path(input.document, suffix=".bin") as doc_path:
                doc_path = str(doc_path)
                # infer()/infer_multi() always mkdir output_path, even when not saving.
                out_dir = tempfile.mkdtemp(prefix="ocr_out_")
                if is_pdf(doc_path):
                    pages = pdf_to_images(doc_path, out_dir)
                    text = self._parse_multi(pages, out_dir)
                else:
                    text = self._parse_single(doc_path, out_dir)
            return ParseDocumentOutput(success=True, text=text or "")
        except Exception as e:
            logger.error(f"unlimited-ocr inference error: {e}", exc_info=True)
            return ParseDocumentOutput(success=False, error=f"infer error: {e}")
