from PIL import Image, ImageDraw
import numpy as np


def detail_panel(folder, destination):
    mask = np.asarray(Image.open(folder / "mask.png")) > 127
    ys, xs = np.where(mask)
    y = int(np.median(ys))
    cx = int(np.median(xs))
    edge = int(np.where(mask[y])[0].min())
    positions = [(cx, y, "central label"), (edge, y, "contour")]
    names = ("input", "plain", "controlled", "composite")
    canvas = Image.new("RGB", (4 * 192, 243), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, name in enumerate(names):
        source = Image.open(folder / (name + ".png")).convert("RGB")
        draw.text((idx * 192 + 4, 3), name, fill="black")
        for i, (x, y, label) in enumerate(positions):
            x = max(48, min(source.width - 48, x))
            y = max(24, min(source.height - 24, y))
            crop = source.crop((x - 48, y - 24, x + 48, y + 24)).resize((192, 96))
            top = 23 + i * 110
            canvas.paste(crop, (idx * 192, top))
            draw.text((idx * 192 + 3, top + 96), label, fill="black")
    canvas.save(destination)
