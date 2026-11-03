"""Magnify the same source-pixel boxes for every rendering of a held-out product."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import numpy as np

REGIONS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "configs" / "inspection-regions.json"
    ).read_text()
)


def detail_panel(folder, destination):
    item = folder.name.split("-")[0]
    mask = np.asarray(Image.open(folder / "mask.png")) > 127
    y = int(np.median(np.nonzero(mask)[0]))
    edge = int(np.flatnonzero(mask[y])[0])
    # A fixed-width boundary patch can extend into background by design.
    edge_x = max(48, min(mask.shape[1] - 48, edge))
    edge_y = max(24, min(mask.shape[0] - 24, y))
    regions = {
        **REGIONS[item],
        "foreground boundary": [edge_x - 48, edge_y - 24, edge_x + 48, edge_y + 24],
    }
    names = ("input", "plain", "controlled", "composite")
    canvas = Image.new("RGB", (4 * 192, 354), "white")
    draw = ImageDraw.Draw(canvas)
    for k, name in enumerate(names):
        with Image.open(folder / (name + ".png")) as source:
            draw.text((k * 192 + 4, 3), name, fill="black")
            for i, (label, box) in enumerate(regions.items()):
                patch = ImageOps.pad(
                    source.crop(box).convert("RGB"),
                    (192, 89),
                    method=Image.Resampling.BICUBIC,
                    color="#f7f6f3",
                )
                top = 22 + i * 110
                canvas.paste(patch, (k * 192, top))
                draw.text((k * 192 + 4, top + 90), label, fill="black")
    canvas.save(destination)
