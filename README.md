# layer-mapping

Paint-order **layer maps** for landscape and Impressionist paintings: which parts of a painting sit in front of
which, split into a few editable layers. Inspired by
[Illustrator's Depth](https://nissmar.github.io/projects/illustrators_depth/) (CVPR 2026), but built only from
openly licensed parts, so it can be released as open source:

* **Backbone:** Depth Anything V2 Small (Apache-2.0)
* **Training data:** procedurally generated painterly landscapes. Exact layer labels, no third-party rights.
* **No** Adobe code, weights or model outputs

Research notes, data and compute plan, and the licensing analysis: [`docs/PLAN.md`](docs/PLAN.md).

## Notebooks (Google Colab)

| Notebook | Purpose | Runtime |
|---|---|---|
| [`01_poc_layer_map.ipynb`](notebooks/01_poc_layer_map.ipynb) | Zero-shot baseline → synthetic data → short fine-tune → compare on real paintings → decision gate | T4, ~45 min |
| [`02_train_full.ipynb`](notebooks/02_train_full.ipynb) | Full training with Drive caching and automatic resume. **Run only after the PoC gate passes.** | A100 / L4, ~3–6 h |

[![Open PoC in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saranunt-0/layer-mapping/blob/main/notebooks/01_poc_layer_map.ipynb)

If the repository is private, add a `GITHUB_TOKEN` secret in Colab (key icon in the sidebar) so the notebook can
clone it.

## Package layout

```
layermap/
  synth.py    procedural painterly landscapes; every stroke writes its layer id, so labels are exact
  data.py     dataset generation (parallel, resumable) and the image/label folder dataset
  model.py    Depth Anything V2 Small loader, preprocessing, prediction (larger value = closer)
  losses.py   scale/shift-invariant L1 + gradient matching (MiDaS) + pairwise ranking loss
  metrics.py  pairwise ordering accuracy: all / local (across edges) / hard (upper pixel in front)
  layers.py   continuous map -> N discrete layers (superpixels + 1-D k-means), RGBA layer export
  train.py    training loop (AMP, cosine LR, checkpoint + resume); also a CLI
```

## Local use

```bash
pip install -e .[dev]
pytest                                   # CPU smoke tests with a tiny random model
python -c "from layermap.data import generate_dataset; generate_dataset('data/train', 2000)"
python -c "from layermap.data import generate_dataset; generate_dataset('data/val', 200, start_seed=1000000)"
python -m layermap.train --train-dirs data/train --val-dir data/val --steps 1500
```

Your own labelled paintings use the same folder layout: `images/<name>.jpg` plus `labels/<name>.png`, a grey-level
map where black = back and brighter = closer.

## License

Code: Apache-2.0 (see `LICENSE`). The Depth Anything V2 Small weights are Apache-2.0 as well. Check the licence of
any extra data before training on it; `docs/PLAN.md` lists what to avoid.
