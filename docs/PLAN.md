# Plan: an open, Illustrator's-Depth-style layer model for paintings

Goal: predict a **paint-order layer map** (which parts of a painting sit in front of which) for landscape and
Impressionist paintings, using only openly licensed components, so the result can be released as open source.

*Researched October 2026. This is not legal advice; see section 3.*

## Summary

* **Illustrator's Depth** (Maruani et al., CVPR 2026; Inria, CityUHK, Adobe Research) predicts a per-pixel
  layer index. It was trained on vector graphics, where the ground truth is the SVG draw order. The code and
  weights were released in May 2026 under the **Adobe Research License**, which allows non-commercial research only.
* You can't get a free model by retraining their recipe. Their training data (**MMSVG-Illustration**) is
  **CC-BY-NC-SA 4.0**, so a model trained on it would carry the same non-commercial restriction.
* **Training on their model's outputs is a real legal problem** if you want an open-source or commercial model.
  Only private, non-commercial research use is covered (section 3).
* The bigger *technical* risk is the **domain gap**. Their model learned flat vector art, and Impressionist
  paintings are almost the opposite. Even their README suggests blurring textured inputs.
* The plan:
  1. Fine-tune **Depth Anything V2 Small** (Apache-2.0, 25M params). It already understands real-world depth.
  2. Train it on **procedurally generated painterly landscapes** whose layer labels are exact by construction.
  3. Snap the output to discrete layers with superpixels.
* Compute is small. The proof of concept fits a free Colab T4 (about 45 min). A full run is roughly
  3–6 A100-hours. Adobe's run was, by my estimate, hundreds of A100-hours.

## 1. What Illustrator's Depth actually is

Checked against the paper's abstract and the official code (`github.com/nissmar/illustrators_depth`).

| Aspect | Finding |
|---|---|
| Definition | Integer layer index per pixel = draw order of the topmost vector shape covering it, dense-ranked. Consecutive shapes with the same colour share an index (`squash_consecutive_same_color: True`). |
| Architecture | Apple **Depth Pro**: two DINOv2 ViT-L/16 encoders (patch + image) and a decoder, FoV head removed, initialised from Depth Pro weights. Input **1536 × 1536**. |
| Loss (released config) | One term: L1 between prediction and target after per-image median shift and mean-absolute-deviation scaling. This is the scale/shift-invariant loss from MiDaS. |
| Data | `OmniSVG/MMSVG-Illustration` (Hugging Face). The training index file is not in the repo. Augmentation: colour jitter, random invert (p=0.3), Gaussian blur. |
| Schedule | Batch 1 per GPU, 40 epochs, cosine LR (encoder 1e-6, decoder/head 5e-6), 192,440 scheduler steps, i.e. 4,811 steps per epoch. |
| Hardware | README: "8 A100 GPUs". `train.py` hard-codes `num_nodes=2` with 8 devices, so 8 or 16 GPUs. That puts the inferred training set at ~38k–77k SVGs. |
| Compute (my estimate) | ~0.6B+ parameters at 1536² for 192k steps on 8–16 A100s: **on the order of several hundred A100-hours**. Not stated by the authors. |
| Release | Code and weights released 05/2026 under the Adobe Research License. Source files also carry an "All Rights Reserved … reproduction strictly forbidden" header. |
| Known weakness | Their README says blurring the input "might help for images with textures (canvas, paper…)". That is exactly the painting case. |

## 2. Licence chain

| Component | License | Commercial / OSS release? | Used here? |
|---|---|---|---|
| Illustrator's Depth code + weights | Adobe Research License | **No** (non-commercial research only, derivatives too) | **No**, not even as a teacher |
| MMSVG-Illustration (their data) | CC-BY-NC-SA 4.0 | **No** | **No** |
| Depth Pro (their backbone) | Apple sample-code license (permissive, no patent grant) | Mostly yes | Not needed |
| Depth Anything V2 **Small** | Apache-2.0 | Yes | **Yes**, the backbone |
| Depth Anything V2 Base / Large / Giant | CC-BY-NC-4.0 | **No** | No |
| Synthetic painterly data (`layermap/synth.py`) | Yours | Yes | **Yes** |
| Museum open-access images (AIC, Met, Cleveland, …) | CC0 | Yes | Test set and hand-labelled data |
| Openclipart vector art | Public domain | Yes | Phase 2 option |
| WikiArt dataset | Non-commercial terms | No | No |

## 3. Can you train on Illustrator's Depth outputs?

**Short answer: not if you want an open-source or commercial model.** It is only defensible for private,
non-commercial research, and anything trained that way inherits the non-commercial restriction.

Why:

1. **Generating the labels is itself "use" of the licensed materials.** The license allows use only for
   "noncommercial research purposes", and it excludes "development of commercial products". If the goal is a
   product or a permissively licensed model, the breach happens when you generate the labels, before anyone asks
   whether the student model is a derivative work.
2. **Derivatives must stay non-commercial.** The license lets you license modifications and derivative works only
   on terms that "limit … to noncommercial research purposes only". Whether a distilled model is a "derivative work"
   under copyright law is unsettled. That uncertainty is risk for you, not protection. The license is also
   *revocable* and terminates automatically on breach.
3. **"Outputs aren't copyrightable" doesn't save you.** Machine-generated depth maps probably have weak or no
   copyright protection (in the US, works need human authorship). But Adobe's leverage is the license terms
   on *using* the model, not copyright in the outputs.
4. **Double non-commercial layer.** Their weights were trained on CC-BY-NC-SA data, so distilling them also
   distils a non-commercial dataset.
5. **Practical exposure.** Publishing "Apache-2.0" weights distilled from a non-commercial model misstates the
   licence chain to everyone downstream. Hosting platforms act on takedown requests.

What *is* reasonable:

* Running their model privately to **benchmark** your own model, for non-commercial research. Don't redistribute
  those outputs or train on them.
* Implementing the **ideas** from the public paper and earlier literature with your own code. Ideas, metrics and
  loss formulas aren't protected by copyright; their code is. This repo was written clean-room: no Adobe code is
  copied, and the loss and metrics cite MiDaS and Depth-in-the-Wild instead.
* If this ever becomes commercial, have a lawyer check two things: your jurisdiction, and whether Adobe holds
  **patents** on the method. A code license doesn't cover patents, and clean-room code doesn't avoid them.

## 4. The main technical risk: domain gap

* Their supervision is flat vector shapes. Impressionist paintings have broken colour, visible strokes, soft or
  lost edges, and canvas texture.
* An Impressionist painting has no ground-truth layer order. Painters work largely alla prima. For **landscapes**,
  though, paint order is close to back-to-front: sky, distance, middle ground, foreground. So the target is about
  *quantised relative depth plus occlusion*.
* Where paint order and physical depth disagree: clouds painted over the sky, reflections on water, flowers over a
  field, figures over a meadow. These are the cases where fine-tuning should help.
* **Metric pitfall found while building this.** On landscapes, the trivial rule "lower in the frame = closer"
  scores **0.92** pairwise ordering accuracy on the synthetic set. Plain accuracy is therefore misleading. The code
  reports `acc_local` (pairs across boundaries) and `acc_hard` (the upper pixel is in front, where that rule
  scores 0). Judge models on those two.

## 5. Data plan

| Tier | What | Size | License | Status |
|---|---|---|---|---|
| A | Synthetic painterly landscapes, exact paint-order labels | PoC 2k; full 30–50k | Yours | Built (`layermap/synth.py`, ~0.25 s/image/core) |
| B | Hand-labelled real paintings, **eval only** | 20–50 | CC0 images + your labels | You: ~5–10 min per painting (estimate) |
| C | Hand-labelled real paintings, training | 100–300 | CC0 images + your labels | Only if the PoC shows a domain gap |
| D | Permissive layered vector art rendered with draw-order labels (Openclipart PD, Noto Emoji Apache-2.0, Fluent Emoji MIT) | 10–50k | As listed | Phase 2; needs an SVG renderer (not built) |

Labelling format for B/C: a grey-level PNG with the same file name as the painting. Black = back, brighter =
closer, same grey = same layer. Any editor works (Krita, GIMP, Photopea). Use a hard-edged brush with
anti-aliasing off, because blended edge pixels would count as extra layers.

Avoid: MMSVG, WikiArt, Adobe Stock, Illustrator's Depth outputs, Depth Anything V2 Base/Large.

## 6. Compute plan

| Stage | Hardware | Work | Estimated time | Cost |
|---|---|---|---|---|
| PoC (`01_poc_layer_map.ipynb`) | Colab T4 (free) | 2.2k images; 1,500 steps × batch 8 @ 518 px | ~45 min total | Free tier |
| Full (`02_train_full.ipynb`) | Colab A100 | 30k images; 30,000 steps × batch 16 @ 518 px | ~3–6 h (measure `s/step` in the PoC) | ~20–90 compute units ≈ $2–9 |
| Data generation | CPU | 30k images at 576 px | ~10 min on 12 cores, ~1 h on 2 | — |

Notes:

* Colab compute-unit burn rates aren't published. Third-party reports for an A100 range from 5 to 15 CU/h, and pay-as-you-go
  is $9.99 per 100 CU. Check before buying.
* Compared with the original, this runs about two orders of magnitude cheaper. Two reasons: a 25M-parameter
  backbone instead of ~0.6B+, and 518 px instead of 1536 px. The trade-off is coarser edges, offset by superpixel
  snapping and higher-resolution inference (770 or 1036 px).
* The PoC log prints `s/step`. Full-run hours ≈ `s/step × steps / 3600`.

## 7. Roadmap and decision gates

1. **Phase 0, now:** run the PoC and label 20+ real paintings. **Gate:** on the real set, fine-tuned beats
   zero-shot on `acc_local` and `acc_hard` by a clear margin. With ~20 paintings, treat gaps under ~2 points as noise.
2. **Phase 1:** full training (notebook 02). Only after the gate passes.
3. **Phase 2:** data realism. Add a draw-order renderer for permissive SVG art, more stroke styles and compositions
   (receding rivers, bridges, haystacks, buildings), and hand-labelled training paintings.
4. **Phase 3:** product features. Fill occluded regions (inpainting) for parallax, PSD export, tiled
   high-resolution inference.

## 8. Verified vs estimated

* **Verified:**
  * from the official repo: architecture, loss, config, data source, licenses (Adobe, MMSVG, Apple)
  * from the upstream README: Depth Anything V2 licensing
  * by CPU tests here: generator, losses, metrics, training/resume loop and both notebooks, on a tiny
    random-init model with substituted paths
* **Estimated or unverified:**
  * Adobe's training set size (inferred from step counts) and GPU-hours
  * Colab timings and costs
  * labelling time
  * the notebooks against the real Depth Anything checkpoint and the AIC API. Neither was reachable from the
    build environment, so the first real run happens in Colab.

## Sources

* Paper: [arXiv 2511.17454](https://arxiv.org/abs/2511.17454) ·
  [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Maruani_Illustrators_Depth_Monocular_Layer_Index_Prediction_for_Image_Decomposition_CVPR_2026_paper.html) ·
  [project page](https://nissmar.github.io/projects/illustrators_depth/)
* Official code and license: [nissmar/illustrators_depth](https://github.com/nissmar/illustrators_depth)
* Training data license: [OmniSVG README](https://github.com/OmniSVG/OmniSVG) (MMSVG: CC-BY-NC-SA 4.0)
* Backbone used there: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
* Backbone used here: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Small: Apache-2.0)
* Losses and metrics: Ranftl et al., *MiDaS*, TPAMI 2020; Chen et al., *Single-Image Depth Perception in the
  Wild*, NeurIPS 2016; Bell et al., *Intrinsic Images in the Wild*, SIGGRAPH 2014
* Open data: [Openclipart FAQ](https://openclipart.org/faq);
  [Art Institute of Chicago API](https://api.artic.edu/docs/)
* Colab costs (third-party, unofficial): [gpuperhour.com](https://gpuperhour.com/blog/google-colab-alternatives),
  [Spheron](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/)
