from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("ERROR: expected the MagicQuill application directory")
        return 2

    app_dir = Path(sys.argv[1]).resolve()
    required = [
        app_dir / "gradio_run.py",
        app_dir / "gradio_magicquill-0.0.1-py3-none-any.whl",
        app_dir / "models/checkpoints/SD1.5/realisticVisionV60B1_v51VAE.safetensors",
        app_dir / "models/controlnet/color_finetune.safetensors",
        app_dir / "models/controlnet/control_v11p_sd15_scribble.safetensors",
        app_dir / "models/inpaint/brushnet/random_mask_brushnet_ckpt/diffusion_pytorch_model.safetensors",
        app_dir / "models/inpaint/brushnet/segmentation_mask_brushnet_ckpt/diffusion_pytorch_model.safetensors",
        app_dir / "models/llava-v1.5-7b-finetune-clean/model-00001-of-00003.safetensors",
        app_dir / "models/llava-v1.5-7b-finetune-clean/model-00002-of-00003.safetensors",
        app_dir / "models/llava-v1.5-7b-finetune-clean/model-00003-of-00003.safetensors",
        app_dir / "models/preprocessor/sk_model.pth",
        app_dir / "models/preprocessor/sk_model2.pth",
        app_dir / "models/clip-vit-large-patch14-336/config.json",
        app_dir / "models/clip-vit-large-patch14-336/preprocessor_config.json",
        app_dir / "models/clip-vit-large-patch14-336/pytorch_model.bin",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ERROR: required files are missing:")
        for path in missing:
            print(f"  {path}")
        return 1

    sys.path.insert(0, str(app_dir))
    sys.path.insert(0, str(app_dir / "MagicQuill" / "LLaVA"))

    import torch
    import diffusers
    import gradio
    import transformers
    import gradio_magicquill
    from MagicQuill import folder_paths

    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot access the NVIDIA GPU")
        return 1

    checkpoints = folder_paths.get_filename_list("checkpoints")
    if not checkpoints:
        print("ERROR: no MagicQuill checkpoints were detected")
        return 1

    import json

    llava_config = json.loads(
        (app_dir / "models/llava-v1.5-7b-finetune-clean/config.json").read_text(encoding="utf-8")
    )
    expected_clip = str((app_dir / "models/clip-vit-large-patch14-336").resolve())
    if llava_config.get("mm_vision_tower") != expected_clip:
        print("ERROR: LLaVA is not configured to use the local CLIP encoder")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} / CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Gradio: {gradio.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Diffusers: {diffusers.__version__}")
    print(f"Checkpoints detected: {len(checkpoints)}")
    print("CLIP vision encoder: local")
    print("INSTALL VERIFY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
