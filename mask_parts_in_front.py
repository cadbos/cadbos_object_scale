import torch

from .common import _resize_mask


class MaskPartsInFront:
    """Фильтрует найденные предметы перед вычитанием из объекта (MaskSubtractParts):
    внутри объекта оставляет только пиксели, которые по карте глубины заметно
    ближе к камере, чем сам объект. Вне объекта маска не меняется."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("parts", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "parts": ("MASK", {"tooltip": "Найденные предметы (узел 319)."}),
            "base": ("MASK", {"tooltip": "Выбранный объект (выход mask узла 400)."}),
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "quantile": ("FLOAT", {"default": 0.98, "min": 0.5, "max": 1.0, "step": 0.01,
                                   "tooltip": "Какая часть объекта считается его ближним краем."}),
            "margin": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.5, "step": 0.01,
                                 "tooltip": "Насколько предмет должен быть ближе ближнего края "
                                            "объекта, чтобы считаться стоящим перед ним."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        }}

    def run(self, parts, base, depth, quantile, margin, depth_near_is_bright):
        p = parts.float()
        if p.dim() == 3:
            p = p.amax(dim=0)
        H, W = p.shape
        dev = p.device
        b = base.float().to(dev)
        if b.dim() == 3:
            b = b.amax(dim=0)
        b = _resize_mask(b, H, W) > 0.5
        pm = p > 0.5
        if not b.any() or not pm.any():
            return (p[None], "объект или предметы пустые — без изменений")

        d = depth[0].float().to(dev)
        if d.dim() == 3:
            d = d.mean(dim=-1)
        d = _resize_mask(d, H, W)
        if not depth_near_is_bright:
            d = 1.0 - d
        near_edge = float(torch.quantile(d[b], float(quantile)))
        inside = pm & b
        in_front = inside & (d > near_edge + float(margin))
        out = (pm & ~b) | in_front
        n_in = int(inside.sum())
        n_front = int(in_front.sum())
        info = (f"пикселей предметов на объекте: {n_in}; "
                f"из них перед объектом (оставлено): {n_front}; "
                f"исключено: {n_in - n_front}")
        return (out.float()[None], info)
