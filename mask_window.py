import torch
import numpy as np

from .common import _blur
from .mask_pick_instance import _split


class MaskWindow:
    """Ограничивает маску «окном» — прямоугольником в долях исходной сцены.
    objects — оставить предметы, которые в основном лежат в окне (целиком);
    clip — просто обрезать маску по окну."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("mask", "window", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("FLOAT", {"forceInput": True, "tooltip": "Левый край окна (узел 423)."}),
                "y": ("FLOAT", {"forceInput": True, "tooltip": "Верхний край окна (узел 422)."}),
                "width": ("FLOAT", {"forceInput": True, "tooltip": "Ширина окна (узел 421)."}),
                "height": ("FLOAT", {"forceInput": True, "tooltip": "Высота окна (узел 420)."}),
                "mode": (["objects", "clip"], {
                    "default": "objects",
                    "tooltip": "objects — оставить предметы, лежащие в окне, целиком; "
                               "clip — обрезать маску по окну."}),
                "min_inside": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Какая доля предмета должна быть в окне (режим objects)."}),
                "separate": ("INT", {
                    "default": 6, "min": 0, "max": 64,
                    "tooltip": "Разделение касающихся предметов, как в Mask Pick Instance."}),
                "feather": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "Мягкий край окна, px (режим clip)."}),
            },
            "optional": {
                "original": ("IMAGE", {"tooltip": "Исходная сцена (узел 1). Нужна, если маска "
                                                  "построена на расширенном кадре."}),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True}),
            },
        }

    def run(self, mask, x, y, width, height, mode, min_inside, separate, feather,
            original=None, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dev = m.device
        x0 = min(max(float(x), 0.0), 1.0)
        y0 = min(max(float(y), 0.0), 1.0)
        x1 = min(max(float(x) + float(width), 0.0), 1.0)
        y1 = min(max(float(y) + float(height), 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:
            z = torch.zeros((1, H, W), device=dev)
            return (z, z, "окно нулевого размера — маска пустая")

        # пересчёт в координаты расширенного кадра (Object Edge Pad)
        if original is not None:
            OH, OW = int(original.shape[1]), int(original.shape[2])
            PW = OW + int(pad_left) + int(pad_right)
            PH = OH + int(pad_top) + int(pad_bottom)
            x0, x1 = (x0 * OW + pad_left) / PW, (x1 * OW + pad_left) / PW
            y0, y1 = (y0 * OH + pad_top) / PH, (y1 * OH + pad_top) / PH
        px0, px1 = int(round(x0 * W)), int(round(x1 * W))
        py0, py1 = int(round(y0 * H)), int(round(y1 * H))
        win = torch.zeros((H, W), device=dev)
        win[py0:py1, px0:px1] = 1.0
        box = f"окно: x {px0}–{px1}, y {py0}–{py1} из {W}x{H}"

        if mode == "clip":
            soft = _blur(win, int(feather)) if feather > 0 else win
            return ((m * soft)[None], soft[None], box)

        a = (m > 0.5).cpu().numpy()
        w_np = np.zeros((H, W), dtype=bool)
        w_np[py0:py1, px0:px1] = True
        keep = np.zeros((H, W), dtype=bool)
        total = kept = 0
        if a.any():
            lab = _split(a, int(separate), None, 0.0)
            for i in np.unique(lab):
                if i == 0:
                    continue
                c = lab == i
                total += 1
                if float((c & w_np).sum()) >= float(min_inside) * float(c.sum()):
                    keep |= c
                    kept += 1
        out = m * torch.from_numpy(keep.astype(np.float32)).to(dev)
        info = f"{box}; предметов найдено: {total}, в окне: {kept}"
        return (out[None], win[None], info)
