# Handoff

Current milestone: the author notebook example and full product experiment completed on RTX 4060 Laptop, PyTorch 2.4.1+cu121. Sixteen model generations and eight foreground composites are saved in results/formal. Four held-out requests contribute to the primary result table. The foreground-controlled model improved average requested hue coverage here but did not improve mean foreground error.

- Original upstream reproduction: python -m src.upstream_example; settings, output hashes and actual deltas against the author's saved images are in results/upstream/run.json.
- Formal generation: python run.py reproduce. Independent saved-pixel verification: python run.py evaluate. Fixed source and model checksums: configs/yahoo-files.json.
- python -m unittest discover -s tests passes. Set VISION_MODEL_CACHE to the parent of yahoo-background/ if reusing weights. Run python -m src.prepare_examples to rebuild the four public inputs and masks consumed by P13.
- Independent GitHub clone at commit 28047ab6: fresh environment ran the GPU demo and all eight paired requests, followed by offline evaluation. All 24 predictions and 16 inputs/masks were byte-identical. All metric deltas were zero. Full audit: results/clean-clone.json. To reproduce again, follow the README and use a separate output directory.


