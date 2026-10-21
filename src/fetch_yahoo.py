"""Materialize the author's immutable files with resumable, hash-checked transfers."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import requests

REPO = "yahoo-inc/photo-background-generation"
REVISION = "719f1ac136b76609d8bd371606d0eaeda731b0dd"
CODE = "a6f12d57031b9cc3dbae54dcb0d2f553ce803265"
ROOT = Path(__file__).resolve().parents[1]
CACHE = (
    Path(os.environ.get("VISION_MODEL_CACHE", str(ROOT / ".cache")))
    / "yahoo-background"
    / REVISION
)
CHUNK = 8 * 1024 * 1024


def url_for(name):
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def write_chunk(task):
    path, url, start, end, total = task
    if path.exists() and path.stat().st_size == end - start + 1:
        return
    for attempt in range(4):
        try:
            with requests.get(
                url,
                params={"segment": start},
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(20, 60),
            ) as r:
                r.raise_for_status()
                if r.headers.get("Content-Range", "") != f"bytes {start}-{end}/{total}":
                    raise IOError("Response range differs from request")
                temp = path.with_suffix(".partial")
                with temp.open("wb") as out:
                    for data in r.iter_content(1024 * 1024):
                        out.write(data)
                if temp.stat().st_size != end - start + 1:
                    raise IOError("Truncated range")
                temp.replace(path)
            return
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise
            time.sleep(1 + 2 * attempt)


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    entries = json.loads((ROOT / "configs" / "yahoo-files.json").read_text())
    for item in entries:
        name = item["rfilename"]
        dest = CACHE / name
        if dest.exists() and dest.stat().st_size == item["size"]:
            if "lfs" in item:
                with dest.open("rb") as handle:
                    valid = (
                        hashlib.file_digest(handle, "sha256").hexdigest()
                        == item["lfs"]["sha256"]
                    )
            else:
                data = dest.read_bytes()
                valid = (
                    hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                    == item["blobId"]
                )
            if valid:
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "lfs" not in item:
            r = requests.get(url_for(name), timeout=45)
            r.raise_for_status()
            blob_hash = hashlib.sha1(
                f"blob {len(r.content)}\0".encode() + r.content
            ).hexdigest()
            if len(r.content) != item["size"] or blob_hash != item["blobId"]:
                raise IOError(name)
            dest.write_bytes(r.content)
            continue
        total = item["size"]
        chunks = dest.with_suffix(".segments")
        chunks.mkdir(exist_ok=True)
        jobs = [
            (
                chunks / f"{start:012d}.part",
                url_for(name),
                start,
                min(start + CHUNK, total) - 1,
                total,
            )
            for start in range(0, total, CHUNK)
        ]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_chunk, jobs))
        out = dest.with_suffix(".assembling")
        digest = hashlib.sha256()
        with out.open("wb") as target:
            for chunk, *_ in jobs:
                buf = chunk.read_bytes()
                digest.update(buf)
                target.write(buf)
        if digest.hexdigest() != item["lfs"]["sha256"]:
            raise ValueError(f"Unexpected SHA-256: {name}")
        out.replace(dest)
        for part, *_ in jobs:
            part.unlink()
        chunks.rmdir()
        print(f"Verified {name}: {total:,} bytes", flush=True)
    code = CACHE / "pipeline_controlnet_inpaint.py"
    if not code.exists():
        response = requests.get(
            f"https://raw.githubusercontent.com/yahoo/photo-background-generation/{CODE}/pipeline_controlnet_inpaint.py",
            timeout=30,
        )
        response.raise_for_status()
        code.write_bytes(response.content)
    if (
        hashlib.sha256(code.read_bytes()).hexdigest()
        != "0f22ebc99a006bab9678bf9d3a339b18cfd3ac9c037c30aeff55f277f8c13a7d"
    ):
        raise ValueError("Upstream source SHA-256 mismatch")
    return CACHE


if __name__ == "__main__":
    download()
