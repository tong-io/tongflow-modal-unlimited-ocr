"""Modal deploy entry for unlimited-ocr.

Document-to-text (OCR / unlimited-horizon document parsing) with Baidu's
``baidu/Unlimited-OCR-2`` model, running on a GPU.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import ast
import io
import logging
import re
import tempfile
from pathlib import Path

import modal
from tongflow import deploy

from tongflow.models.asset import Asset
from tongflow.models.parse_document import ParseDocumentInput, ParseDocumentOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
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

# Grounding markup the model interleaves with the Markdown, boxes in 0..999
# page coordinates: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>, or
# the bare form <|det|>label [x1, y1, x2, y2]<|/det|>.
_GROUNDING = re.compile(
    r"<\|ref\|>(?P<ref>.*?)<\|/ref\|><\|det\|>(?P<box>.*?)<\|/det\|>"
    r"|<\|det\|>\s*(?P<label>[A-Za-z_][\w-]*)\s*(?P<bbox>\[[^\]]+\])\s*<\|/det\|>",
    re.DOTALL,
)
FIGURE_LABEL = "image"
FIGURE_JPEG_QUALITY = 90
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
        "tongflow==1.0.0", "fastapi[standard]",
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


def _boxes(text: str) -> list[list[float]]:
    """Parse a det payload (one box or a list of boxes) without eval."""
    try:
        v = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return []
    if v and isinstance(v[0], (int, float)):
        v = [v]
    return [list(b) for b in v if isinstance(b, (list, tuple)) and len(b) == 4]


def crop_figure(page, box: list[float]) -> bytes | None:
    """Crop a 0..999-normalized box out of a PIL page image as JPEG bytes."""
    w, h = page.size
    x1, y1, x2, y2 = (int(c / 999 * s) for c, s in zip(box, (w, h, w, h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    buf = io.BytesIO()
    page.crop((x1, y1, x2, y2)).save(buf, format="JPEG", quality=FIGURE_JPEG_QUALITY)
    return buf.getvalue()


def split_pages(raw: str, page_paths: list[str], figures: list[Asset]) -> list[str]:
    """Split raw model output on <PAGE> into per-page Markdown (as upstream's result.md).

    Figure regions are cropped from their page image and appended to
    ``figures``; each leaves ``![image N](#image-N)`` in the text, N being its
    1-based position in ``figures``. Other grounding markup is dropped.
    """
    from PIL import Image

    pages = raw.split(PAGE_MARKER)[1:] if PAGE_MARKER in raw else [raw]
    out: list[str] = []
    for i, page in enumerate(pages):
        path = page_paths[i] if i < len(page_paths) else None
        loaded: dict[str, object] = {}

        def replace(m: re.Match[str]) -> str:
            label = (m.group("ref") or m.group("label") or "").strip()
            if label != FIGURE_LABEL or path is None:
                return ""
            if "img" not in loaded:
                loaded["img"] = Image.open(path).convert("RGB")
            links: list[str] = []
            for box in _boxes(m.group("box") or m.group("bbox") or ""):
                data = crop_figure(loaded["img"], box)
                if data is None:
                    continue
                n = len(figures) + 1
                figures.append(asset(data, mime="image/jpeg", filename=f"image-{n}.jpg"))
                links.append(f"![image {n}](#image-{n})")
            return "\n\n".join(links)

        page = _GROUNDING.sub(replace, page)
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

    def _parse(self, image_paths: list[str], out_dir: str) -> tuple[str, list[Asset]]:
        # No prompt: the model builds its trained "<PAGE><image>" * n +
        # "Multi page parsing." layout, and crop_mode=None picks gundam tiling
        # for a single page and base mode for several.
        ngram_window = SINGLE_NGRAM_WINDOW if len(image_paths) == 1 else MULTI_NGRAM_WINDOW
        pages: list[str] = []
        figures: list[Asset] = []
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
            pages.extend(split_pages(raw or "", batch, figures))
        return "\n\n".join(p for p in pages if p), figures

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
                text, figures = self._parse(pages, out_dir)
            return ParseDocumentOutput(success=True, text=text, images=figures or None)
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
