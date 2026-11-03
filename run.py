"""Compare Yahoo ControlNet with its zero-scale and hard-composite controls."""

import argparse
import csv
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from src.data import ROOT, items, prepare
from src.fetch_yahoo import CODE, REVISION, download


def metrics(original, edited, foreground, color):
    x = np.asarray(original.convert("RGB"), dtype=float)
    y = np.asarray(edited.convert("RGB"), dtype=float)
    err = np.abs(x - y).mean(axis=2)
    mask = np.asarray(foreground) > 127
    if not mask.any() or mask.all():
        raise ValueError("Need foreground and background pixels")
    edge = (np.asarray(foreground.filter(ImageFilter.MaxFilter(7))) > 0) ^ (
        np.asarray(foreground.filter(ImageFilter.MinFilter(7))) > 0
    )
    axis = {"red": 0, "blue": 2}[color]
    other = np.delete(y, axis, axis=2).max(axis=2)
    # Dominant requested hue outside the foreground is a narrow, auditable prompt proxy.
    return {
        "foreground_mae": float(err[mask].mean()),
        "boundary_mae": float(err[edge].mean()),
        "target_hue_fraction": float(
            ((y[:, :, axis] - other > 16) & (~mask)).sum() / (~mask).sum()
        ),
    }


def evaluate(out):
    records = json.loads((out / "manifest.json").read_text())
    rows = []
    if (
        records["status"] != "completed"
        or len({c["case"] for c in records["cases"]}) != records["expected_cases"]
    ):
        raise ValueError(
            "Generation is incomplete; resume or rerun before reporting scores"
        )
    for case in records["cases"]:
        sub = out / case["case"]
        with Image.open(sub / "input.png") as img, Image.open(sub / "mask.png") as mask:
            for method in ("plain", "controlled", "composite"):
                with Image.open(sub / (method + ".png")) as result:
                    rows.append(
                        {
                            "id": case["id"],
                            "split": case["split"],
                            "color": case["color"],
                            "method": method,
                            **metrics(img, result, mask, case["color"]),
                        }
                    )
    with (out / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in ("plain", "controlled", "composite"):
        ds = [r for r in rows if r["method"] == method and r["split"] == "test"]
        if ds:
            summary.append(
                {
                    "method": method,
                    "cases": len(ds),
                    **{
                        k: float(np.mean([r[k] for r in ds]))
                        for k in (
                            "foreground_mae",
                            "boundary_mae",
                            "target_hue_fraction",
                        )
                    },
                }
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    subset = [r for r in rows if r["method"] == "controlled" and r["split"] == "test"]
    if subset:
        from src.panels import detail_panel

        for item in sorted({r["id"] for r in subset}):
            detail_panel(out / (item + "-red"), out / ("details-" + item + ".jpg"))
        worst = min(subset, key=lambda r: r["target_hue_fraction"])
        detail_panel(
            out / (worst["id"] + "-" + worst["color"]), out / "lowest-scene-score.jpg"
        )
    return rows


def load_model():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the verified fp16 configuration")
    path = download()
    spec = importlib.util.spec_from_file_location(
        "yahoo_pipeline", path / "pipeline_controlnet_inpaint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe = module.StableDiffusionControlNetInpaintPipeline.from_pretrained(
        str(path), torch_dtype=torch.float16, local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run(out, demo):
    import torch
    import diffusers

    pipe = load_model()
    out.mkdir(parents=True, exist_ok=True)
    examples = [i for i in items() if i["id"] != "B083V9C8D2"]
    params = {"size": 384, "steps": 12, "seed": 13}
    records = {
        "status": "in_progress",
        "model": REVISION,
        "source": CODE,
        "config": params,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(),
        "cases": [],
        "expected_cases": 1 if demo else 2 * len(examples),
    }
    for item in examples[:1] if demo else examples:
        img, mask, digest = prepare(item, params["size"])
        hole = ImageOps.invert(mask)
        for color in ("red",) if demo else ("red", "blue"):
            case = item["id"] + "-" + color
            prompt = (
                f"Product photo of {item['name']} on a plain {color} studio background"
            )
            sub = out / case
            sub.mkdir(exist_ok=True)
            img.save(sub / "input.png")
            mask.save(sub / "mask.png")
            features = {}
            for method, scale in (("plain", 0.0), ("controlled", 1.0)):
                gen = torch.Generator(device="cuda").manual_seed(params["seed"])
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        image=img,
                        mask_image=hole,
                        control_image=hole.convert("RGB"),
                        height=img.height,
                        width=img.width,
                        num_inference_steps=params["steps"],
                        generator=gen,
                        controlnet_conditioning_scale=scale,
                    ).images[0]
                torch.cuda.synchronize()
                features[method] = {
                    "seconds": time.perf_counter() - start,
                    "cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                }
                result.save(sub / (method + ".png"))
            with Image.open(sub / "plain.png") as plain:
                Image.composite(img, plain, mask).save(sub / "composite.png")
            records["cases"].append(
                {
                    "id": item["id"],
                    "split": item["split"],
                    "case": case,
                    "prompt": prompt,
                    "color": color,
                    "sha256": digest,
                    "timings": features,
                }
            )
            (out / "manifest.json").write_text(json.dumps(records, indent=2))
    records["status"] = "completed"
    (out / "manifest.json").write_text(json.dumps(records, indent=2))
    evaluate(out)
    example = next(
        (k for k in records["cases"] if k["split"] == "test"), records["cases"][0]
    )
    p = out / example["case"]
    panel = Image.new("RGB", (4 * 192, 276), "white")
    draw = ImageDraw.Draw(panel)
    with Image.open(p / "mask.png") as m:
        x0, y0, x1, y1 = m.getbbox()
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    for i, name in enumerate(("input", "plain", "controlled", "composite")):
        with Image.open(p / (name + ".png")) as im:
            panel.paste(im.resize((192, 192)), (192 * i, 20))
            crop = im.crop((cx - 48, cy - 16, cx + 48, cy + 16)).resize((192, 64))
            panel.paste(crop, (192 * i, 212))
            draw.text((192 * i + 4, 3), name, fill="black")
    panel.save(out / "comparison.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["demo", "reproduce", "evaluate"])
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    output = opts.output or ROOT / "results" / (
        "demo" if opts.action == "demo" else "formal"
    )
    if opts.action == "evaluate":
        evaluate(output)
    else:
        run(output, opts.action == "demo")
