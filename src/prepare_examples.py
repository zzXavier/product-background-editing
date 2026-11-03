"""Export the public product image and mask bundle consumed by the editing-agent experiment."""

import hashlib
import json

try:
    from .data import ROOT, items, prepare
except ImportError:
    from data import ROOT, items, prepare


def main():
    target = ROOT / "examples" / "products"
    target.mkdir(parents=True, exist_ok=True)
    cases = []
    for item in items():
        if item["id"] == "B083V9C8D2":
            continue
        image, mask, source_hash = prepare(item)
        case = target / item["id"]
        case.mkdir(exist_ok=True)
        image.save(case / "input.png")
        mask.save(case / "mask.png")
        cases.append(
            {
                "id": item["id"],
                "split": item["split"],
                "name": item["name"],
                "size": 384,
                "source_path": item["photo"],
                "original_sha256": source_hash,
                "files": {
                    f'{item["id"]}/{name}': hashlib.sha256(
                        (case / name).read_bytes()
                    ).hexdigest()
                    for name in ["input.png", "mask.png"]
                },
                "requests": [
                    f"Product photo of {item['name']} on a plain {color} studio background"
                    for color in ["red", "blue"]
                ],
            }
        )
    (target / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
