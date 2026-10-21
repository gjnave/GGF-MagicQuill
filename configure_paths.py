from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("ERROR: expected the MagicQuill application directory")
        return 2

    app_dir = Path(sys.argv[1]).resolve()
    config_path = app_dir / "models/llava-v1.5-7b-finetune-clean/config.json"
    clip_path = app_dir / "models/clip-vit-large-patch14-336"

    if not config_path.is_file() or not clip_path.is_dir():
        print("ERROR: model files are incomplete; cannot configure local paths")
        return 1

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mm_vision_tower"] = str(clip_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Configured local CLIP encoder: {clip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
