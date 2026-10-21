from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("ERROR: expected the MagicQuill application directory")
        return 2

    app_dir = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(app_dir))
    sys.path.insert(0, str(app_dir / "MagicQuill" / "LLaVA"))

    import torch
    from MagicQuill.llava_new import LLaVAModel
    from MagicQuill.scribble_color_edit import ScribbleColorEditModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    print(f"Loading LLaVA on {torch.cuda.get_device_name(0)}...")
    llava_model = LLaVAModel()
    print("LLaVA loaded")

    print("Loading the MagicQuill image-editing pipeline...")
    edit_model = ScribbleColorEditModel()
    print("Image-editing pipeline loaded")

    del edit_model
    del llava_model
    torch.cuda.empty_cache()
    print("MODEL SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
