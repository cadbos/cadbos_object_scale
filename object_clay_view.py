import torch

from .common import _resize_mask, _blur, _bbox, _push_pull_fill
from .mask_convex_hull import MaskConvexHull


VIEWS = {
    "front": "seen from the front: its front faces the camera",
    "back": "seen from behind: its back faces the camera and its front faces away "
            "from the camera. If it has a backrest, we see the back side of the "
            "backrest, which is closest to the camera and hides most of the seat",
    "front-left": "seen at three-quarters from the front: its front faces the camera "
                  "and is turned toward the left edge of the image",
    "front-right": "seen at three-quarters from the front: its front faces the camera "
                   "and is turned toward the right edge of the image",
    "back-left": "seen at three-quarters from behind: its back faces the camera and its "
                 "front is turned away from the camera, toward the left edge of the image",
    "back-right": "seen at three-quarters from behind: its back faces the camera and its "
                  "front is turned away from the camera, toward the right edge of the image",
    "facing-left": "seen exactly from the side, in profile: its front points to the "
                   "left edge of the image",
    "facing-right": "seen exactly from the side, in profile: its front points to the "
                    "right edge of the image",
}


# ключ ракурса -> направление камеры для Multiple-Angles LoRA
AZIMUTH = {
    "front": "front view",
    "front-left": "front-left quarter view",
    "front-right": "front-right quarter view",
    "facing-left": "left side view",
    "facing-right": "right side view",
    "back": "back view",
    "back-left": "back-left quarter view",
    "back-right": "back-right quarter view",
}


# порядок важен: сначала составные варианты, потом простые
_PARSE = [
    ("back-left", ["back-left", "left-back"]),
    ("back-right", ["back-right", "right-back"]),
    ("front-left", ["front-left", "left-front"]),
    ("front-right", ["front-right", "right-front"]),
    ("back", ["back", "behind", "rear"]),
    ("front", ["front"]),
    ("facing-left", ["left"]),
    ("facing-right", ["right"]),
]


def _norm(text):
    return str(text or "").lower().replace("_", "-").replace(" ", "-").replace("–", "-")


def _parse_view(text):
    t = _norm(text)
    for key, words in _PARSE:
        if any(w in t for w in words):
            return key
    return None


def _last_key(text, table):
    """Ключ из table, чьё слово встречается в тексте последним."""
    best, pos = None, -1
    for key, words in table:
        for w in words:
            i = text.rfind(w)
            if i > pos:
                best, pos = key, i
    return best


def _after_answer(text):
    """Если модель написала 'ANSWER: ...', оставить только то, что после последнего 'answer'."""
    i = text.rfind("answer")
    return text[i + len("answer"):] if i >= 0 else text


def _combine_view(side_text, dir_text):
    """Ответ «какая сторона видна» + ответ «куда смотрит перед» -> ключ VIEWS."""
    s = _after_answer(str(side_text or "").lower())
    d = _after_answer(str(dir_text or "").lower())
    for phrase in ("away from the camera", "away from the viewer", "away from us"):
        d = d.replace(phrase, "away")
    side = _last_key(s, [("back", ["back", "behind"]), ("front", ["front"])])
    direction = _last_key(d, [("left", ["left"]), ("right", ["right"]),
                              ("away", ["away"]),
                              ("camera", ["camera", "toward the viewer", "towards the viewer"])])
    if direction == "away":      # перед смотрит от камеры — видна спинка
        return "back"
    if direction == "camera":    # перед смотрит в камеру — виден перед
        return "front"
    if direction in ("left", "right"):
        if side == "front":
            return f"front-{direction}"
        return f"facing-{direction}"   # «back» или нет ответа — профиль, ошибка менее заметна
    return side


class ObjectClayView:
    """Рисует старый объект как серую глиняную модель по карте глубины
    (ближе — светлее, с освещением сверху слева) и выдаёт фразу о ракурсе."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "text", "info", "camera")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "mask": ("MASK", {"tooltip": "Старый объект (узел 322)."}),
            "view": (["auto", "none"] + list(VIEWS), {
                "default": "auto",
                "tooltip": "Какой стороной объект повёрнут к камере. auto — по ответам "
                           "Qwen3-VL (view_side + view_dir, иначе view_text); "
                           "none — без фразы, только глиняная модель."}),
            "relief": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                 "tooltip": "Сила объёма. 0 — только яркость по глубине."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "view_text": ("STRING", {"forceInput": True,
                                     "tooltip": "Ответ Qwen3-VL одним словом: front, back-left ..."}),
            "view_side": ("STRING", {"forceInput": True,
                                     "tooltip": "Ответ Qwen3-VL: front или back."}),
            "view_dir": ("STRING", {"forceInput": True,
                                    "tooltip": "Ответ Qwen3-VL: left, right, camera или away."}),
            "simplify": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                                   "tooltip": "Упростить модель до грубого блока, чтобы Qwen брал "
                                              "из неё только позу, а не форму. 0 — точная модель."}),
            "elevation": (["eye-level shot", "elevated shot", "high-angle shot", "low-angle shot"], {
                "default": "elevated shot",
                "tooltip": "Высота камеры для LoRA. Для фото интерьера обычно elevated shot."}),
            "distance": (["medium shot", "close-up", "wide shot"], {"default": "medium shot"}),
            "mirror_lr": ("BOOLEAN", {
                "default": False,
                "tooltip": "Поменять left/right в промпте LoRA, если поворот получается зеркальным."}),
        }}

    def run(self, depth, mask, view, relief, depth_near_is_bright,
            view_text="", view_side=None, view_dir=None, simplify=0.0,
            elevation="elevated shot", distance="medium shot", mirror_lr=False):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dev = m.device
        img = torch.ones((H, W, 3), device=dev)
        obj = (m > 0.5).float()
        bb = _bbox(obj)
        if bb is None:
            return (img[None], "", "маска пустая", "")
        size = float(max(bb[2] - bb[0], bb[3] - bb[1]))

        d = depth[0].float().to(dev)
        if d.dim() == 3:
            d = d.mean(dim=-1)
        d = _resize_mask(d, H, W)
        if not depth_near_is_bright:
            d = 1.0 - d
        vals = d[obj > 0.5]
        lo, hi = torch.quantile(vals, 0.02), torch.quantile(vals, 0.98)
        dn = ((d - lo) / (hi - lo).clamp_min(1e-4)).clamp(0, 1)

        if simplify > 0:
            # грубый блок: выпуклая оболочка, глубина в промежутках — по самому предмету
            hull = MaskConvexHull().run(obj[None])[0][0].to(dev)
            gap = (hull - obj).clamp(0, 1)
            dn = _push_pull_fill(dn[..., None], gap, exclude=(1 - obj))[..., 0]
            obj = hull
            dn = _blur(dn, max(2, int(round(float(simplify) * size * 0.08))))
        else:
            dn = _blur(dn, 2)

        # нормали поверхности по глубине и простое освещение сверху слева
        gy, gx = torch.gradient(dn)
        k = float(relief) * size * 0.5
        n = torch.stack([-gx * k, -gy * k, torch.ones_like(dn)], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True)
        light = torch.tensor([-0.45, -0.55, 0.7], device=dev)
        light = light / light.norm()
        lam = (n * light).sum(-1).clamp_min(0)
        shade = (0.18 + 0.55 * lam + 0.22 * dn).clamp(0, 1)

        a = _blur(obj, max(1, int(round(float(simplify) * size * 0.02))))[..., None]
        img = img * (1 - a) + shade[..., None] * a

        if view == "none":
            return (img[None], "", "ракурс: без фразы (только глиняная модель)", "")
        if view != "auto":
            key, src = view, "вручную"
        elif view_side is not None or view_dir is not None:
            key = _combine_view(view_side, view_dir)
            src = (f"по ответам VLM: сторона '{str(view_side).strip()}', "
                   f"направление '{str(view_dir).strip()}'")
        else:
            key = _parse_view(view_text)
            src = f"по ответу VLM '{str(view_text).strip()}'"
        if key is None:
            return (img[None], "", f"ракурс не определён ({src}) — без фразы", "")
        text = (f"Required view of the result: the object is {VIEWS[key]}. "
                f"In the gray clay model, lighter parts are closer to the "
                f"camera and darker parts are farther away.")
        az = AZIMUTH[key]
        if mirror_lr:
            az = az.replace("left", "@").replace("right", "left").replace("@", "right")
        camera = f"<sks> {az} {elevation} {distance}"
        return (img[None], text, f"ракурс: {key} ({src}); камера: {camera}", camera)
