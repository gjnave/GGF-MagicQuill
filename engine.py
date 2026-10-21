"""Original MagicQuill editing pipeline shared by local and phone interfaces."""
import base64
import gc
import io
from pathlib import Path
import sys
import threading
import time
import uuid

from PIL import Image, ImageOps
import numpy as np

APP = Path(__file__).resolve().parent
sys.path[:0] = [str(APP / "MagicQuill"), str(APP / "MagicQuill/LLaVA")]


def read_image(value):
    if not isinstance(value, str) or "," not in value:
        raise ValueError("Upload an image and draw the area to edit first.")
    with Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))) as image:
        return ImageOps.exif_transpose(image).copy()


def image_data(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.stage = "Ready."
        self.started = 0
        self.step = self.total = None
        self.resolution = 512
        self.editor = self.llava = None

    def begin(self, text):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("A job is already running. Wait or stop it first.")
        self.busy = True
        self.started = time.monotonic()
        self.stage = text
        self.step = self.total = None

    def release_models(self):
        import torch
        import comfy.model_management as mm
        mm.unload_all_models()
        self.editor = self.llava = None
        gc.collect()
        torch.cuda.empty_cache()

    def finish(self):
        try:
            self.release_models()
        finally:
            self.busy = False
            self.step = self.total = None
            self.lock.release()

    def generate(self, value, checkpoint, negative, fine, grow, edge, color, strength,
                 seed, steps, cfg, sampler, scheduler):
        self.begin("Loading the image-editing models…")
        try:
            import torch
            from MagicQuill.scribble_color_edit import ScribbleColorEditModel
            import comfy.utils
            import comfy.model_management as mm
            mm.interrupt_current_processing(False)
            def report(current, total, preview=None):
                self.step, self.total = current, total
                self.stage = f"Generating · step {current} of {total}"
            comfy.utils.set_progress_bar_global_hook(report)
            data = value.get("from_frontend", {}) if isinstance(value, dict) else {}
            original = read_image(data.get("original_image")).convert("RGB")
            def tensor(image):
                return torch.from_numpy(np.asarray(image.convert("RGB")).astype(np.float32) / 255)[None]
            def mask(encoded):
                if not encoded:
                    return torch.zeros((1, original.height, original.width), dtype=torch.float32)
                image = read_image(encoded)
                if image.size != original.size:
                    raise ValueError("The image and painted selection do not match. Reload the image and paint again.")
                alpha = np.asarray(image.convert("RGBA").getchannel("A")).astype(np.float32) / 255
                return torch.from_numpy(1.0 - alpha)[None]
            selection = mask(data.get("total_mask"))
            if not torch.any(selection):
                raise ValueError("Draw on the part of the image you want to change first.")
            original_tensor = tensor(original)
            colored = tensor(read_image(data["add_color_image"])) if data.get("add_color_image") else original_tensor
            add, remove = mask(data.get("add_edge_image")), mask(data.get("remove_edge_image"))
            prompt = value.get("from_backend", {}).get("prompt", "")
            if torch.sum(remove).item() > 0 and torch.sum(add).item() == 0:
                prompt = prompt or "empty scene"
                edge /= 3
            self.editor = ScribbleColorEditModel()
            self.stage = "Preparing your edit…"
            seed = int(seed) if int(seed) >= 0 else int.from_bytes(__import__("os").urandom(4), "little")
            with torch.no_grad():
                _, result, _, _ = self.editor.process(checkpoint, original_tensor, colored,
                    prompt, negative, selection, add, remove, int(grow), "enable", fine,
                    float(edge), float(color), float(strength), seed, int(steps), float(cfg), sampler, scheduler)
            picture = Image.fromarray((result.squeeze(0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
            output = APP / "outputs" / (time.strftime("magicquill-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8] + ".png")
            output.parent.mkdir(exist_ok=True)
            picture.save(output)
            value.setdefault("from_backend", {})["generated_image"] = image_data(picture).split(",", 1)[1]
            self.stage = "Finished. Saved in outputs."
            return value, str(output)
        except Exception as error:
            self.stage = f"Edit stopped: {error}"
            raise
        finally:
            self.finish()

    def guess(self, data):
        self.begin("Reading your drawing…")
        try:
            import torch
            from MagicQuill.llava_new import LLaVAModel
            def tensor(image):
                return torch.from_numpy(np.asarray(image.convert("RGB")).astype(np.float32) / 255)[None]
            original = read_image(data["original_image"])
            color = read_image(data["add_color_image"]) if data.get("add_color_image") else original
            if data.get("add_edge_image"):
                alpha = np.asarray(read_image(data["add_edge_image"]).convert("RGBA").getchannel("A")).astype(np.float32) / 255
                add = torch.from_numpy(1.0 - alpha)[None]
            else:
                add = torch.zeros((1, original.height, original.width))
            self.llava = LLaVAModel()
            with torch.no_grad():
                _, first, second = self.llava.process(tensor(original), tensor(color), add)
            self.stage = "Suggested prompt ready."
            return ", ".join(text for text in (first, second) if text)
        finally:
            self.finish()


ENGINE = Engine()
