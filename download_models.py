"""Download the previously validated MagicQuill model set, with resume and hashes."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from configure_paths import main as configure
import sys

APP = Path(__file__).resolve().parent


def main():
    manifest = json.loads((APP / "model_manifest.json").read_text())
    for index, item in enumerate(manifest, 1):
        target = APP / "models" / item["path"]
        print(f"[{index}/{len(manifest)}] {item['path']}", flush=True)
        if not target.exists() or (item["sha256"] and target.stat().st_size != item["size"]):
            staging = APP / "models/.downloads" / item["repo"].replace("/", "--")
            downloaded = Path(hf_hub_download(item["repo"], item["filename"],
                              revision=item["revision"], local_dir=staging))
            if downloaded.stat().st_size != item["size"]:
                raise RuntimeError(f"Incomplete download: {item['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
        if item["sha256"]:
            with target.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256") if hasattr(hashlib, "file_digest") else None
                if digest is None:
                    digest = hashlib.sha256()
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Verification failed: {item['path']}. File preserved.")
    sys.argv = [sys.argv[0], str(APP)]
    if configure():
        raise RuntimeError("Local model configuration failed.")
    print("All MagicQuill models downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
