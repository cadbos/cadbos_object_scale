import os
import sys
import types
import importlib

import numpy as np
import torch
from PIL import Image

from .common import _resize_mask, _bbox

_OA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "orient_anything_v2")
_OA_NAMES = ("utils", "vggt", "vision_tower")   # имена модулей репозитория (у ComfyUI тоже есть utils)
_STATE = {"model": None, "au": None}

# направление переда предмета на картинке, шаг 45° (0° = перёд к камере)
_KEYS = ["front", "front-right", "facing-right", "back-right",
         "back", "back-left", "facing-left", "front-left"]
# ключ -> команда для Multiple-Angles LoRA (её «право/лево» — со стороны камеры)
_LORA = {
    "front": "front view",
    "front-right": "front-left quarter view",
    "facing-right": "left side view",
    "back-right": "back-left quarter view",
    "back": "back view",
    "back-left": "back-right quarter view",
    "facing-left": "right side view",
    "front-left": "front-right quarter view",
}


def _is_oa(name):
    return name in _OA_NAMES or any(name.startswith(n + ".") for n in _OA_NAMES)


def _load():
    """Загружает Orient Anything V2 один раз. Модули репозитория импортируются
    в изоляции, чтобы его пакет utils не столкнулся с utils самого ComfyUI."""
    if _STATE["model"] is not None:
        return _STATE["model"], _STATE["au"]
    if not os.path.isdir(_OA_DIR):
        raise RuntimeError(f"Orient Anything V2 не найден в {_OA_DIR}")
    saved = {k: v for k, v in list(sys.modules.items()) if _is_oa(k)}
    for k in saved:
        del sys.modules[k]
    if "rembg" not in sys.modules:          # rembg не нужен: фон убираем по своей маске
        try:
            import rembg  # noqa: F401
        except Exception:
            sys.modules["rembg"] = types.ModuleType("rembg")
    sys.path.insert(0, _OA_DIR)
    try:
        vt = importlib.import_module("vision_tower")
        au = importlib.import_module("utils.app_utils")
        pa = importlib.import_module("utils.paths")
    finally:
        if _OA_DIR in sys.path:
            sys.path.remove(_OA_DIR)
        for k in [k for k in list(sys.modules) if _is_oa(k)]:
            del sys.modules[k]
        sys.modules.update(saved)

    import folder_paths
    ckpt_dir = os.path.join(folder_paths.models_dir, "orient_anything")
    ckpt = os.path.join(ckpt_dir, pa.HF_CKPT_PATH)
    if not os.path.exists(ckpt):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id="Viglong/OriAnyV2_ckpt", filename=pa.HF_CKPT_PATH,
                               repo_type="model", local_dir=ckpt_dir)
    dtype = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    model = vt.VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    _STATE.update(model=model, au=au)
    return model, au


def _elevation_phrase(el):
    if el < -15:
        return "low-angle shot"
    if el < 20:
        return "eye-level shot"
    if el < 45:
        return "elevated shot"
    return "high-angle shot"


class ObjectOrientAnything:
    """Определяет ракурс объекта на сцене моделью Orient Anything V2
    и выдаёт команду для Multiple-Angles LoRA. Работает для любых предметов."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("view", "camera", "elevation", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Сцена (узел 3)."}),
            "mask": ("MASK", {"tooltip": "Старый объект (узел 322)."}),
            "flip_lr": ("BOOLEAN", {"default": False,
                                    "tooltip": "Поменять лево/право, если повороты выходят зеркальными."}),
            "distance": (["medium shot", "close-up", "wide shot"], {"default": "medium shot"}),
            "keep_on_gpu": ("BOOLEAN", {"default": False,
                                        "tooltip": "Держать модель в видеопамяти между запусками "
                                                   "(быстрее, но ~5 ГБ VRAM постоянно)."}),
        }}

    def run(self, image, mask, flip_lr, distance, keep_on_gpu):
        img = image[0].float()[..., :3]
        H, W, _ = img.shape
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        m = (_resize_mask(m.to(img.device), H, W) > 0.5).float()
        bb = _bbox(m)
        if bb is None:
            return ("", "", "eye-level shot", "маска пустая")
        x0, y0, x1, y1 = bb
        rgb = (img[y0:y1, x0:x1].clamp(0, 1) * 255).byte().cpu().numpy()
        alpha = (m[y0:y1, x0:x1] * 255).byte().cpu().numpy()
        pil = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

        model, au = _load()
        pil = au.resize_foreground(pil, 0.85)
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        mm.free_memory(6 * 1024 ** 3, dev)   # освободить место: модель занимает ~5 ГБ
        model.to(dev)
        try:
            with torch.no_grad():
                ans = au.inf_single_case(model, pil, None)
        finally:
            if not keep_on_gpu:
                model.to("cpu")
                mm.soft_empty_cache()

        az = float(ans["ref_az_pred"])
        el = float(ans["ref_el_pred"])
        sym = int(float(ans["ref_alpha_pred"]))
        elev = _elevation_phrase(el)
        nums = f"азимут {az:.0f}°, возвышение {el:.0f}°, симметрия {sym}"
        if sym == 0:
            return ("", "", elev, f"{nums} — у предмета нет выраженного переда, поворот не нужен")
        key = _KEYS[int(((az + 22.5) % 360) // 45)]
        if flip_lr:
            key = key.replace("left", "@").replace("right", "left").replace("@", "right")
        camera = f"<sks> {_LORA[key]} elevated shot {distance}"
        return (key, camera, elev, f"{nums} → ракурс {key}; камера: {camera}; "
                                   f"высота камеры для 2-го прохода: {elev}")
