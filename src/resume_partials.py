"""Recover complete range chunks left by an interrupted sequential checkpoint download."""

from fetch_yahoo import CACHE, CHUNK


def recover():
    total = 0
    for src in sorted(CACHE.rglob("*.safetensors.tmp")):
        dest = src.with_suffix("")
        if dest.exists():
            continue
        chunkdir = dest.with_suffix(".segments")
        chunkdir.mkdir(exist_ok=True)
        with src.open("rb") as inp:
            complete = src.stat().st_size // CHUNK
            for i in range(complete):
                start = CHUNK * i
                path = chunkdir / f"{start:012d}.part"
                if path.exists():
                    continue
                inp.seek(start)
                data = inp.read(CHUNK)
                if len(data) == CHUNK:
                    temp = path.with_suffix(".recover")
                    temp.write_bytes(data)
                    temp.replace(path)
                    total += len(data)
    print(f"Recovered {total:,} bytes; final model SHA-256 remains mandatory.")


if __name__ == "__main__":
    recover()
