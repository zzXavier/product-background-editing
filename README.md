# Product background editing

Run Yahoo's CVPR workshop foreground-conditioned inpainting checkpoint and compare it with the identical model at control scale zero and a hard foreground composite. Results use the same product, prompt, seed and inference budget. The one-image command is `python run.py demo`.

## Setup and reproduction

Use Python 3.11, install CUDA PyTorch 2.4.1, then `python -m pip install -r requirements.txt`. A CUDA GPU and enough disk space for the multi-GB float32 checkpoint are required. The first run downloads hashed weight files and the pinned upstream pipeline into ignored `.cache/`. Set VISION_MODEL_CACHE to share the model snapshot.

1. `python -m unittest discover -s tests`
2. `python run.py demo` runs one public product photograph with three comparisons.
3. `python run.py reproduce` runs both studio colors on two development and two held-out products.
4. `python run.py evaluate` recomputes scores and the held-out summary from saved PNGs, without loading a model.

`src/data.py` provides attribution-linked ABO input preparation. `src/fetch_yahoo.py` downloads upstream code and weights without reimplementing the model. `run.py` adds paired runs, segmentation, hard compositing, edge and foreground error, reproducible labels, and reporting. The central enlargement exposes packaging text and logo changes visually. Foreground MAE is in RGB byte units; boundary MAE includes legitimate backdrop changes. The background score is a predeclared hue test for a plain red/blue studio prompt. It measures color compliance only. Test subjects never participate in parameter selection.

See HANDOFF.md for current experiment status, links to versioned results and limitations. The extracted mask can miss low-contrast white packaging. The two-bottle conditioner catalog entry is retained for provenance and excluded from experiments because the foreground method assumes one connected product.


To reproduce the actual author notebook example rather than the product benchmark, run python -m src.upstream_example. It extracts the saved swan image and foreground mask from the immutable notebook itself, uses the original bedroom prompt, 512 px, seed 13, 20 steps and control scales 0 and 1. The report logs MAE versus the two embedded author notebook outputs and local output hashes; bitwise equality is not expected for fp16.
