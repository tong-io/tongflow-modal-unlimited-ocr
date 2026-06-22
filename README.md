<img width="1582" height="977" alt="截屏2026-06-22 21 48 19" src="https://github.com/user-attachments/assets/4cb8c45b-4dfa-49be-a134-af5a84f7dfde" />


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

## Acknowledgements

This plugin is a thin TongFlow wrapper around Baidu's **Unlimited-OCR** model. All credit for the model goes to the Unlimited-OCR team. The weights are released under the MIT License.

- Model: [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (Hugging Face)
- Source: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (GitHub)

## Citation

If you use this plugin, please cite the upstream Unlimited-OCR work. A formal
citation is forthcoming from the authors (see the [upstream repo](https://github.com/baidu/Unlimited-OCR#citation)); until then, reference it as:

```bibtex
@misc{unlimitedocr2026,
  title  = {Unlimited OCR Works},
  author = {Baidu},
  year   = {2026},
  howpublished = {\url{https://github.com/baidu/Unlimited-OCR}}
}
```
