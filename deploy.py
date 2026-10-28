"""Modal deploy entry for unlimited-ocr.

Document-to-text (OCR / unlimited-horizon document parsing) with Baidu's
``baidu/Unlimited-OCR-2`` model, running on a GPU.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import modal
from tongflow import deploy

from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset_as_path
from tongflow.slots import current_params, node_slot


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v

# Per-run knobs offered under the node's collapsed "Advanced" section.
# Pure literal (the platform scanner reads it by AST, never imports this
# module). Values reach the handlers via current_params(); an untouched
# control is absent there and falls back to the plugin default.
TONGFLOW_SLOT_PARAMS = {
    "parse-document": {
        "pdf_dpi": {"type": "select", "options": [150, 200, 300], "default": 300, "label": "PDF render DPI"},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Plugin-internal config; not exposed via ABI.
REPO_ID = "baidu/Unlimited-OCR-2"
MODEL_DIR = f"/models/{REPO_ID}"
PDF_DPI = 300
# Total sequence (page image tokens + generated text) per `infer` call; the
# model's own default and its max_position_embeddings.
MAX_LENGTH = 131072
NO_REPEAT_NGRAM_SIZE = 35
# Upstream evaluation setup: a narrow n-gram window for one page, a wide one
# for multi-page documents.
SINGLE_NGRAM_WINDOW = 128
MULTI_NGRAM_WINDOW = 1024
# SR-SWA keeps per-step decode cost flat, but the Transformers path still caps
# one call at MAX_LENGTH tokens. A page costs ~273 image tokens (base mode,
# 1024px) plus its text, so ~48 pages leave ~2.4k output tokens per page.
# Longer PDFs are parsed in page windows of this size, joined in order.
PAGES_PER_CALL = 48

# Grounding markup the model interleaves with the Markdown:
# <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
_REF_DET = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
_BARE_DET = re.compile(r"<\|det\|>\s*[A-Za-z_][\w-]*\s*\[[^\]]+\]\s*<\|/det\|>", re.DOTALL)
PAGE_MARKER = "<PAGE>"

volume = modal.Volume.from_name("models", create_if_missing=True)

image = (
    # devel image: the model's Triton sparse-attention kernels need a C compiler.
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "tongflow==0.3.4", "fastapi[standard]",
        # Upstream-tested stack for the remote modeling code.
        "transformers==4.52.1",
        "tokenizers==0.21.4",
        "huggingface-hub==0.36.0",
        "einops==0.8.2",
        "addict==2.4.0",
        "easydict==1.13",
        "pymupdf==1.27.2.2",
        "requests",
        "pillow",
        "numpy",
        "tqdm",
    )
    # expandable_segments reduces allocator fragmentation on the long
    # prefill + KV cache of multi-page OCR.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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


def clean_pages(raw: str) -> list[str]:
    """Split raw model output on <PAGE> and drop grounding markup (as upstream's result.md)."""
    pages = raw.split(PAGE_MARKER)[1:] if PAGE_MARKER in raw else [raw]
    out: list[str] = []
    for page in pages:
        page = _BARE_DET.sub("", _REF_DET.sub("", page))
        page = page.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
        out.append(page.strip())
    return out


@deploy
@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=3600,
    secrets=[secrets],
    scaledown_window=2,
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

    def _parse(self, image_paths: list[str], out_dir: str) -> str:
        # No prompt: the model builds its trained "<PAGE><image>" * n +
        # "Multi page parsing." layout, and crop_mode=None picks gundam tiling
        # for a single page and base mode for several.
        ngram_window = SINGLE_NGRAM_WINDOW if len(image_paths) == 1 else MULTI_NGRAM_WINDOW
        pages: list[str] = []
        for start in range(0, len(image_paths), PAGES_PER_CALL):
            batch = image_paths[start : start + PAGES_PER_CALL]
            raw = self.model.infer(
                self.tokenizer,
                image_file=batch,
                output_path=out_dir,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                ngram_window=ngram_window,
                eval_mode=True,
            )
            pages.extend(clean_pages(raw or ""))
        return "\n\n".join(p for p in pages if p)

    @modal.method()
    @node_slot(NodeSlots.PARSE_DOCUMENT)
    def parse_document(self, input: ParseDocumentInput) -> ParseDocumentOutput:
        if input.document is None:
            return ParseDocumentOutput(success=False, error="Missing `document` Asset")
        try:
            with asset_as_path(input.document, suffix=".bin") as doc_path:
                doc_path = str(doc_path)
                out_dir = tempfile.mkdtemp(prefix="ocr_out_")
                if is_pdf(doc_path):
                    pages = pdf_to_images(doc_path, out_dir, dpi=_adv("pdf_dpi", PDF_DPI))
                else:
                    pages = [doc_path]
                text = self._parse(pages, out_dir)
            return ParseDocumentOutput(success=True, text=text)
        except Exception as e:
            logger.error(f"unlimited-ocr inference error: {e}", exc_info=True)
            return ParseDocumentOutput(success=False, error=f"infer error: {e}")

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
