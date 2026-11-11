import torch
import torch.nn.functional as F

from .common import _resize_mask, _bbox


class ReferenceScaleForCrop:
    """Готовит Picture 2 для Qwen: объект с референса на белом квадрате, такого
    размера, какой ожидается в кадре этапа 1 (по высоте рамки старого объекта)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "reference": ("IMAGE",),
            "reference_mask": ("MASK",),
            "box_mask": ("MASK", {"tooltip": "Рамка объекта этапа 1 (узел 155, object_mask)."}),
            "region_mask": ("MASK", {"tooltip": "Маска перерисовки этапа 1 (узел 93)."}),
            "context": ("FLOAT", {
                "default": 1.25, "min": 1.0, "max": 3.0, "step": 0.05,
                "tooltip": "Должно совпадать с context_from_mask_extend_factor узла 100."}),
            "size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
        },
        "optional": {
            "target_width": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True,
                                     "tooltip": "Ширина кропа этапа 1 (GetImageSize узла 3)."}),
            "target_height": ("INT", {"default": 0, "min": 0, "max": 16384, "forceInput": True,
                                      "tooltip": "Высота кропа этапа 1 (GetImageSize узла 3)."}),
        }}

    def run(self, reference, reference_mask, box_mask, region_mask, context,
            size, target_width=0, target_height=0):
        ref = reference[0].float()[..., :3]
        dev = ref.device
        RH, RW, _ = ref.shape
        rm = _resize_mask(reference_mask[0].float().to(dev), RH, RW).clamp(0, 1)
        bb = _bbox((rm > 0.5).float())
        box = (box_mask[0].float().to(dev) > 0.5).float()
        bbb = _bbox(box)
        reg = (region_mask[0].float().to(dev) > 0.01).float()
        rbb = _bbox(reg)
        if bb is None or bbb is None or rbb is None:
            return (reference, "пустая маска — референс передан без изменений")
        rx0, ry0, rx1, ry1 = bb
        obj = ref[ry0:ry1, rx0:rx1]
        am = rm[ry0:ry1, rx0:rx1]
        oh, ow = ry1 - ry0, rx1 - rx0

        # кроп этапа 1 (как считает InpaintCropImproved) и доля рамки в нём
        reg_w, reg_h = rbb[2] - rbb[0], rbb[3] - rbb[1]
        IH, IW = reg.shape[0], reg.shape[1]
        if target_width > 0 and target_height > 0:  # кроп с пропорциями кадра
            A = float(target_width) / float(target_height)
            cw, ch = reg_w * float(context), reg_h * float(context)
            if cw / max(1.0, ch) < A:
                cw = ch * A
            else:
                ch = cw / A
            if cw > IW:
                cw, ch = float(IW), IW / A
            if ch > IH:
                ch, cw = float(IH), IH * A
            CW, CH = int(target_width), int(target_height)
        else:  # квадратный кроп
            ch = min(max(reg_w, reg_h) * float(context), float(max(IH, IW)))
            CW = CH = int(size)
        f = (bbb[3] - bbb[1]) / max(1.0, ch)
        f = min(0.95, max(0.05, f))

        th = f * CH
        tw = ow * th / oh
        if tw > 0.95 * CW:  # слишком широкий объект — вписать по ширине
            tw = 0.95 * CW
            th = oh * tw / ow
        th, tw = max(2, int(round(th))), max(2, int(round(tw)))
        aa = th < oh
        o = F.interpolate(obj.permute(2, 0, 1)[None], size=(th, tw), mode="bicubic",
                          align_corners=False, antialias=aa)[0].permute(1, 2, 0).clamp(0, 1)
        a = F.interpolate(am[None, None], size=(th, tw), mode="bilinear",
                          align_corners=False, antialias=aa)[0, 0].clamp(0, 1)[..., None]
        canvas = torch.ones((CH, CW, 3), device=dev)
        y0 = (CH - th) // 2
        x0 = (CW - tw) // 2
        canvas[y0:y0 + th, x0:x0 + tw] = canvas[y0:y0 + th, x0:x0 + tw] * (1 - a) + o * a
        info = (f"доля высоты рамки в кадре этапа 1: {f:.3f}; "
                f"объект на референсе: {tw}x{th}px на {CW}x{CH}")
        return (canvas[None], info)
