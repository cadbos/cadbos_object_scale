import torch

from .common import _blur


class MaskSmoothEdge:
    """Сглаживает контур маски: убирает мелкие зубцы и выступы, не меняя
    общей формы. Детали уже примерно radius px пропадают."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "radius": ("INT", {"default": 3, "min": 0, "max": 32,
                               "tooltip": "Сила сглаживания, px. 0 — без изменений."}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05,
                                    "tooltip": "Меньше 0.5 — контур чуть шире, больше — чуть уже."}),
        },
        "optional": {
            "soft": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "Ширина мягкого перехода по краю. 0 — резкий край. "
                                          "0.15 — сглаженный край ~2 px."}),
        }}

    def run(self, mask, radius, threshold, soft=0.0):
        m = mask.float()
        if m.dim() == 2:
            m = m[None]
        out = []
        for i in range(m.shape[0]):
            s = (m[i] > 0.5).float()
            if radius > 0:
                s = _blur(_blur(s, int(radius)), int(radius))  # почти гауссово размытие
            if soft > 0:
                s = ((s - float(threshold)) / float(soft) + 0.5).clamp(0, 1)
            else:
                s = (s > float(threshold)).float()
            out.append(s)
        return (torch.stack(out),)
