import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask, _blur


class MaskDropUnchanged:
    """Убирает из маски нового объекта куски, которые на картинке совпадают с
    чистым фоном (например, опора стола под стулом): они не часть объекта.
    Убираются только куски у внешнего края маски."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Маска нового объекта (узел 342)."}),
            "image": ("IMAGE", {"tooltip": "Результат этапа 1 (узел 101)."}),
            "clean": ("IMAGE", {"tooltip": "Чистый фон без старого объекта (узел 220)."}),
            "tolerance": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01,
                                    "tooltip": "Насколько пиксель может отличаться от фона, "
                                               "чтобы считаться неизменным."}),
            "min_area": ("INT", {"default": 64, "min": 0, "max": 100000,
                                 "tooltip": "Мелкие неизменные пятна (px) не трогать."}),
        }}

    def run(self, mask, image, clean, tolerance, min_area):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dev = m.device
        img = image[0].float().to(dev)[..., :3]
        bg = clean[0].float().to(dev)[..., :3]
        if img.shape[0] != H or img.shape[1] != W:
            img = F.interpolate(img.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0)
        if bg.shape[0] != H or bg.shape[1] != W:
            bg = F.interpolate(bg.permute(2, 0, 1)[None], size=(H, W), mode="bilinear",
                               align_corners=False)[0].permute(1, 2, 0)
        diff = (img - bg).abs().amax(dim=-1)
        diff = _blur(diff, 2)
        obj = (m > 0.5).cpu().numpy()
        same = ((diff < float(tolerance)).cpu().numpy()) & obj
        same = ndi.binary_opening(same, iterations=2)
        lab, n = ndi.label(same)
        outside = ~obj
        edge = ndi.binary_dilation(outside, iterations=2)
        drop = np.zeros_like(obj)
        removed = 0
        for i in range(1, n + 1):
            c = lab == i
            if int(c.sum()) < int(min_area) or not (c & edge).any():
                continue
            drop |= c
            removed += int(c.sum())
        out = obj & ~drop
        info = f"убрано неизменённых пикселей: {removed} из {int(obj.sum())}"
        return (torch.from_numpy(out.astype(np.float32))[None].to(dev), info)
