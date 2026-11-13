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
        }}

    def run(self, mask, radius, threshold):
        m = mask.float()
        if m.dim() == 2:
            m = m[None]
        if radius <= 0:
            return (m,)
        out = []
        for i in range(m.shape[0]):
            s = (m[i] > 0.5).float()
            s = _blur(_blur(s, int(radius)), int(radius))  # почти гауссово размытие
            out.append((s > float(threshold)).float())
        return (torch.stack(out),)
