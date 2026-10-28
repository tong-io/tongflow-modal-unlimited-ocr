<img width="1582" height="977" alt="截屏2026-06-22 21 48 19" src="https://github.com/user-attachments/assets/4cb8c45b-4dfa-49be-a134-af5a84f7dfde" />


# tongflow-modal-unlimited-ocr

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Document-to-text (OCR / unlimited-horizon document parsing) with Baidu's **Unlimited OCR 2** (`baidu/Unlimited-OCR-2`), running on a GPU via [Modal](https://modal.com). An alternative to `tongflow-modal-docling` and `tongflow-modal-paddle` on the same `parse-document` slot.

## Capabilities

- **Document → text** (`parse-document`) — parse an image or a PDF into Markdown. Single images use the high-detail *gundam* config (cropped tiling); PDFs are rendered to page images and parsed together in *base* config, so the model keeps cross-page context (SR-SWA keeps per-step cost flat as documents grow). Very long PDFs are processed in windows of 48 pages.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin downloads the `baidu/Unlimited-OCR-2` weights to a Modal Volume, deploys to your Modal account automatically, and caches the build. The weights are public — no Hugging Face token required.

## Acknowledgements

This plugin is a thin TongFlow wrapper around Baidu's **Unlimited OCR 2** model. All credit for the model goes to the Unlimited-OCR team. The weights are released under the MIT License.

- Model: [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) (Hugging Face)
- Source: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (GitHub)

## Citation

If you use this plugin, please cite the upstream Unlimited-OCR work:

```bibtex
@misc{yin2026unlimitedocrworks,
  title         = {Unlimited OCR Works},
  author        = {Youyang Yin and Huanhuan Liu and YY and Qunyi Xie and Chaorun Liu and Shiqi Yang and Shaohua Wang and Zhanlong Liu and Hao Zou and Jinyue Chen and Shu Wei and Jingjing Wu and Mingxin Huang and Zhen Wu and Guibin Wang and Tengyu Du and Lei Jia},
  year          = {2026},
  eprint        = {2606.23050},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.23050}
}
```
