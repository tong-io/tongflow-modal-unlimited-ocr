# tongflow-modal-unlimited-ocr

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Document-to-text (OCR / long-horizon document parsing) with Baidu's **Unlimited-OCR** (`baidu/Unlimited-OCR`), running on a GPU via [Modal](https://modal.com). An alternative to `tongflow-modal-docling` and `tongflow-modal-paddle` on the same `parse-document` slot.

## Capabilities

- **Document → text** (`parse-document`) — parse an image or PDF page into Markdown. Single images use the high-detail *gundam* config (cropped tiling); multi-page PDFs are rendered to images and parsed in *base* config.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the `baidu/Unlimited-OCR` weights to a Modal Volume, deploys to your Modal account automatically, and caches the build. The weights are public — no Hugging Face token required.
