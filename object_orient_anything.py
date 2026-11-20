import os
import sys
import types
import importlib
import importlib.util

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


def _preload_pkg(name):
    """Регистрирует пакет репозитория (utils, vggt) под его именем явно.
    Иначе Python возьмёт одноимённый пакет ComfyUI: папка utils в репозитории
    без __init__.py, а «обычный» пакет всегда побеждает такую папку."""
    pkg_dir = os.path.join(_OA_DIR, name)
    init = os.path.join(pkg_dir, "__init__.py")
    if os.path.exists(init):
        spec = importlib.util.spec_from_file_location(name, init,
                                                      submodule_search_locations=[pkg_dir])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = types.ModuleType(name)
        mod.__path__ = [pkg_dir]
        sys.modules[name] = mod


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
        _preload_pkg("utils")
        _preload_pkg("vggt")
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
    model = model.to(dtype)   # bf16: в памяти 2.5 ГБ вместо 5
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


def _infer(pil):
    """Азимут, возвышение и симметрия предмета на картинке (PIL RGB/RGBA)."""
    model, au = _load()
    import comfy.model_management as mm
    dev = mm.get_torch_device()
    mm.free_memory(4 * 1024 ** 3, dev)   # модели в bf16 нужно ~3 ГБ с запасом
    model.to(dev)
    try:
        with torch.no_grad():
            ans = au.inf_single_case(model, pil, None)
    finally:
        if not _STATE.get("keep"):
            model.to("cpu")
            mm.soft_empty_cache()
    return (float(ans["ref_az_pred"]), float(ans["ref_el_pred"]),
            int(float(ans["ref_alpha_pred"])))


def _ang_dist(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _white_bg_pil(image, thr=0.03, ratio=0.85):
    """Картинка предмета на белом фоне -> квадратная вырезка с полями, как в Orient Anything."""
    img = image[0].float()[..., :3].clamp(0, 1)
    fg = (1.0 - img).amax(dim=-1) > thr
    bb = _bbox(fg.float())
    if bb is None:
        crop = img
    else:
        x0, y0, x1, y1 = bb
        crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = int(max(h, w) / ratio)
    canvas = torch.ones((side, side, 3))
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop.cpu()
    return Image.fromarray((canvas.numpy() * 255).astype(np.uint8), "RGB")


class ObjectOrientAnything:
    """Определяет ракурс объекта на сцене моделью Orient Anything V2
    и выдаёт команду для Multiple-Angles LoRA. Работает для любых предметов."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("view", "camera", "elevation", "info", "azimuth")

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
        _STATE["keep"] = bool(keep_on_gpu)
        img = image[0].float()[..., :3]
        H, W, _ = img.shape
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        m = (_resize_mask(m.to(img.device), H, W) > 0.5).float()
        bb = _bbox(m)
        if bb is None:
            return ("", "", "eye-level shot", "маска пустая", -1.0)
        x0, y0, x1, y1 = bb
        rgb = (img[y0:y1, x0:x1].clamp(0, 1) * 255).byte().cpu().numpy()
        alpha = (m[y0:y1, x0:x1] * 255).byte().cpu().numpy()
        pil = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
        _, au = _load()
        pil = au.resize_foreground(pil, 0.85)

        az, el, sym = _infer(pil)
        elev = _elevation_phrase(el)
        nums = f"азимут {az:.0f}°, возвышение {el:.0f}°, симметрия {sym}"
        if sym == 0:
            return ("", "", elev, f"{nums} — у предмета нет выраженного переда, поворот не нужен",
                    -1.0)
        key = _KEYS[int(((az + 22.5) % 360) // 45)]
        if flip_lr:
            key = key.replace("left", "@").replace("right", "left").replace("@", "right")
        camera = f"<sks> {_LORA[key]} elevated shot {distance}"
        return (key, camera, elev, f"{nums} → ракурс {key}; камера: {camera}; "
                                   f"высота камеры для 2-го прохода: {elev}", az)


class ObjectOrientPick:
    """Проверяет моделью Orient Anything, какой из вариантов повёрнутого референса
    (после 1-го и после 2-го прохода LoRA) повёрнут как старый объект, и отдаёт его."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_a": ("IMAGE", {"tooltip": "После 1-го прохода (узел 395)."}),
            "image_b": ("IMAGE", {"tooltip": "После 2-го прохода (выход VAEDecode 2-го прохода)."}),
            "azimuth": ("FLOAT", {"forceInput": True,
                                  "tooltip": "Выход azimuth узла Object Orient Anything."}),
            "prefer_b_within": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 180.0, "step": 5.0,
                                          "tooltip": "Брать 2-й проход (камера на уровне глаз), если он "
                                                     "отличается от нужного поворота не больше, "
                                                     "чем на столько градусов."}),
        }}

    def run(self, image_a, image_b, azimuth, prefer_b_within):
        if azimuth < 0:
            return (image_b, "ракурс старого объекта не определён — взят 2-й проход")
        az_a, _, _ = _infer(_white_bg_pil(image_a))
        az_b, _, _ = _infer(_white_bg_pil(image_b))
        da, db = _ang_dist(az_a, azimuth), _ang_dist(az_b, azimuth)
        info = (f"нужно {azimuth:.0f}°; 1-й проход {az_a:.0f}° (ошибка {da:.0f}°), "
                f"2-й проход {az_b:.0f}° (ошибка {db:.0f}°)")
        if db <= float(prefer_b_within) or db <= da:
            return (image_b, info + " → взят 2-й проход")
        return (image_a, info + " → взят 1-й проход")
