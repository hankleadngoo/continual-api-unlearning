import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parent / ".cache" / "models" / "CodeLlama-7b-hf"
REV = "6c284d1468fe6c413cf56183e69b194dcfa27fe6"
BASE = f"https://huggingface.co/codellama/CodeLlama-7b-hf/resolve/{REV}/"
CHUNK = 32 * 1024 * 1024


def fetch(url, path, start=None, end=None):
    temp = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(10):
        try:
            offset = temp.stat().st_size if start is not None and temp.exists() else 0
            if start is not None and offset == end - start + 1:
                temp.replace(path)
                return
            if start is not None and offset > end - start + 1:
                raise RuntimeError("Temporary part exceeds requested range")
            current = start + offset if start is not None else None
            headers = {} if start is None else {"Range": f"bytes={current}-{end}"}
            request_url = url if start is None else url + ("&" if "?" in url else "?") + f"offset={current}&attempt={attempt}"
            request = urllib.request.Request(request_url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if start is not None:
                    expected = f"bytes {current}-{end}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                        raise RuntimeError("Server did not return requested byte range")
                with temp.open("ab" if offset else "wb") as out:
                    shutil.copyfileobj(response, out, 1024 * 1024)
            if start is not None and temp.stat().st_size != end - start + 1:
                raise RuntimeError("Incomplete byte range")
            temp.replace(path)
            return
        except Exception as error:
            print(f"Retry {attempt + 1}: {path.name}: {error}", flush=True)
            if attempt == 9:
                raise
            time.sleep(min(30, 3 * (attempt + 1)))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    fetch(f"https://huggingface.co/api/models/codellama/CodeLlama-7b-hf/revision/{REV}?blobs=true", ROOT / "download-manifest.json")
    manifest = json.loads((ROOT / "download-manifest.json").read_text())
    names = {"config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "LICENSE", "USE_POLICY.md"}
    files = [f for f in manifest["siblings"] if f["rfilename"] in names or f["rfilename"].endswith(".safetensors")]
    # Parts and assembled weights coexist until their SHA-256 is verified.
    if shutil.disk_usage(ROOT).free < 28 * 1024**3:
        raise RuntimeError("Need at least 28 GiB free for download and verification")
    for info in files:
        name, size = info["rfilename"], info["size"]
        target = ROOT / name
        sha = info.get("lfs", {}).get("sha256")
        if target.exists() and target.stat().st_size == size and (not sha or digest(target) == sha):
            print(f"Verified existing {name}", flush=True)
            continue
        url = BASE + name
        print(f"Downloading {name}: {size / 1024**3:.3f} GiB", flush=True)
        if size < CHUNK:
            fetch(url, target)
        else:
            parts = ROOT / (name + ".parts")
            parts.mkdir(exist_ok=True)
            count = (size + CHUNK - 1) // CHUNK
            def part(i):
                start, end = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
                path = parts / f"{i:05d}"
                if not path.exists() or path.stat().st_size != end - start + 1:
                    # Unique URL avoids intermediary caches serving another range.
                    fetch(url + f"?part={i}", path, start, end)
                return i
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for done, future in enumerate(concurrent.futures.as_completed([pool.submit(part, i) for i in range(count)]), 1):
                    future.result()  # A finished worker may have failed; never count it as success.
                    if done % 25 == 0 or done == count:
                        print(f"{name}: {done}/{count} parts", flush=True)
            print(f"All parts present for {name}; assembling and checking SHA-256", flush=True)
            temporary = target.with_suffix(target.suffix + ".assembling")
            with temporary.open("wb") as out:
                for i in range(count):
                    with (parts / f"{i:05d}").open("rb") as src:
                        shutil.copyfileobj(src, out, 1024 * 1024)
            if temporary.stat().st_size != size or (sha and digest(temporary) != sha):
                raise RuntimeError(f"Checksum failed: {name}; parts retained")
            temporary.replace(target)
            for i in range(count):
                (parts / f"{i:05d}").unlink()
            parts.rmdir()
        if target.stat().st_size != size or (sha and digest(target) != sha):
            raise RuntimeError(f"Verification failed: {name}")
        print(f"Verified {name}", flush=True)
    (ROOT / "DOWNLOAD_COMPLETE").write_text(REV + "\n", encoding="ascii")
    print(f"COMPLETE: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
