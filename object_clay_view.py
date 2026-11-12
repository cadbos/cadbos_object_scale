import torch

from .common import _resize_mask, _blur, _bbox


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


def _parse_view(text):
    t = str(text).lower().replace("_", "-").replace(" ", "-").replace("–", "-")
    for key, words in _PARSE:
        if any(w in t for w in words):
            return key
    return None


class ObjectClayView:
    """Рисует старый объект как серую глиняную модель по карте глубины
    (ближе — светлее, с освещением сверху слева) и выдаёт фразу о ракурсе.
    Нужен вместо плоского серого силуэта (узел 397)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "text", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "mask": ("MASK", {"tooltip": "Старый объект (узел 61)."}),
            "view": (["auto"] + list(VIEWS), {
                "default": "auto",
                "tooltip": "Какой стороной объект повёрнут к камере. auto — взять из "
                           "view_text (ответ Qwen3-VL); если его нет — без фразы."}),
            "relief": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1,
                                 "tooltip": "Сила объёма. 0 — только яркость по глубине."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "view_text": ("STRING", {"forceInput": True,
                                     "tooltip": "Ответ Qwen3-VL: front, back, left ..."}),
        }}

    def run(self, depth, mask, view, relief, depth_near_is_bright, view_text=""):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dev = m.device
        img = torch.ones((H, W, 3), device=dev)
        obj = (m > 0.5).float()
        bb = _bbox(obj)
        if bb is None:
            return (img[None], "", "маска пустая")

        d = depth[0].float().to(dev)
        if d.dim() == 3:
            d = d.mean(dim=-1)
        d = _resize_mask(d, H, W)
        if not depth_near_is_bright:
            d = 1.0 - d
        vals = d[obj > 0.5]
        lo, hi = torch.quantile(vals, 0.02), torch.quantile(vals, 0.98)
        dn = ((d - lo) / (hi - lo).clamp_min(1e-4)).clamp(0, 1)
        dn = _blur(dn, 2)

        # нормали поверхности по глубине и простое освещение сверху слева
        size = float(max(bb[2] - bb[0], bb[3] - bb[1]))
        gy, gx = torch.gradient(dn)
        k = float(relief) * size * 0.5
        n = torch.stack([-gx * k, -gy * k, torch.ones_like(dn)], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True)
        light = torch.tensor([-0.45, -0.55, 0.7], device=dev)
        light = light / light.norm()
        lam = (n * light).sum(-1).clamp_min(0)
        shade = (0.18 + 0.55 * lam + 0.22 * dn).clamp(0, 1)

        a = _blur(obj, 1)[..., None]
        img = img * (1 - a) + shade[..., None] * a

        key = view if view != "auto" else _parse_view(view_text or "")
        if key is None:
            return (img[None], "", f"ракурс не задан (ответ VLM: '{view_text}')")
        text = (f"Required view of the result: the object is {VIEWS[key]}. "
                f"In Picture 2, lighter parts of the gray model are closer to the "
                f"camera and darker parts are farther away.")
        src = "вручную" if view != "auto" else f"по ответу VLM '{str(view_text).strip()}'"
        return (img[None], text, f"ракурс: {key} ({src})")
