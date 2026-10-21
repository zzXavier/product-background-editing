"""Run the actual Yahoo notebook's swan example using its embedded input and segmentation."""

import argparse
import base64
from io import BytesIO
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import requests
from src.fetch_yahoo import ROOT, CODE


def fixtures():
    folder = ROOT / ".cache" / "upstream-example"
    folder.mkdir(parents=True, exist_ok=True)
    note = folder / "inference.ipynb"
    if not note.exists():
        u = f"https://raw.githubusercontent.com/yahoo/photo-background-generation/{CODE}/inference.ipynb"
        r = requests.get(u, timeout=40)
        r.raise_for_status()
        note.write_bytes(r.content)
    notebook = json.loads(note.read_bytes())
    imgs = {}
    for key, idx in [
        ("original", 4),
        ("foreground", 5),
        ("author_plain", 7),
        ("author_controlled", 10),
    ]:
        image = next(
            d["data"]["image/png"]
            for d in notebook["cells"][idx]["outputs"]
            if "image/png" in d.get("data", {})
        )
        data = base64.b64decode(image)
        path = folder / (key + ".png")
        path.write_bytes(data)
        imgs[key] = Image.open(BytesIO(data)).copy()
    return folder, imgs


def main(prepare_only=False):
    folder, im = fixtures()
    if prepare_only:
        print({k: [v.mode, *v.size] for k, v in im.items()})
        return
    import torch
    from run import load_model

    pipe = load_model()
    img = im["original"].convert("RGB")
    hole = ImageOps.invert(im["foreground"].convert("L"))
    info = {
        "checkpoint": "719f1ac136b76609d8bd371606d0eaeda731b0dd",
        "upstream_code": CODE,
        "prompt": "A dark swan in a bedroom",
        "seed": 13,
        "steps": 20,
        "size": list(img.size),
        "dtype": "float16",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "results": {},
    }
    for name, scale in [("plain", 0.0), ("controlled", 1.0)]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=info["prompt"],
                image=img,
                mask_image=hole,
                control_image=hole.convert("RGB"),
                height=img.height,
                width=img.width,
                num_inference_steps=20,
                controlnet_conditioning_scale=scale,
                generator=torch.Generator(device="cuda").manual_seed(13),
                guess_mode=False,
            ).images[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        path = folder / ("local_" + name + ".png")
        result.save(path)
        error = np.abs(
            np.asarray(result, dtype=float)
            - np.asarray(im["author_" + name].convert("RGB"), dtype=float)
        )
        info["results"][name] = {
            "wall_seconds": elapsed,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mae_vs_embedded_author_output": float(error.mean()),
        }
    dest = ROOT / "results" / "upstream"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-only", action="store_true")
    main(p.parse_args().prepare_only)
