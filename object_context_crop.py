import torch
import torch.nn.functional as F

from .common import _resize_mask, _bbox, _draw_rect


class ObjectContextCrop:
    """Кадрирует сцену вокруг объекта с запасом и обводит объект рамкой.
    Для вопросов к VLM: виден весь предмет и окружение, без рваных краёв маски."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")

    COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.3, 1.0)}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Сцена (узел 3)."}),
            "mask": ("MASK", {"tooltip": "Старый объект (узел 61)."}),
            "context": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 4.0, "step": 0.1,
                                  "tooltip": "Размер кадра относительно объекта."}),
            "size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32,
                             "tooltip": "Длинная сторона результата, px."}),
            "thickness": ("INT", {"default": 4, "min": 1, "max": 32}),
            "color": (["red", "green", "blue"], {"default": "red"}),
        }}

    def run(self, image, mask, context, size, thickness, color):
        img = image[0].float()[..., :3]
        H, W, _ = img.shape
        dev = img.device
        m = mask.float().to(dev)
        if m.dim() == 3:
            m = m.amax(dim=0)
        m = _resize_mask(m, H, W)
        bb = _bbox(m)
        if bb is None:
            return (image, "маска пустая — передана вся сцена")
        x0, y0, x1, y1 = bb
        side = max(x1 - x0, y1 - y0) * float(context)
        cw, ch = int(round(min(W, side))), int(round(min(H, side)))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cx0 = int(round(min(max(cx - cw / 2.0, 0), W - cw)))
        cy0 = int(round(min(max(cy - ch / 2.0, 0), H - ch)))
        crop = img[cy0:cy0 + ch, cx0:cx0 + cw]

        s = float(size) / max(cw, ch)
        nw, nh = max(2, int(round(cw * s))), max(2, int(round(ch * s)))
        crop = F.interpolate(crop.permute(2, 0, 1)[None], size=(nh, nw), mode="bicubic",
                             align_corners=False, antialias=s < 1)[0].permute(1, 2, 0)
        crop = crop.clamp(0, 1).contiguous()
        pad = int(thickness) + 3
        crop = _draw_rect(crop,
                          int(round((x0 - cx0) * s)) - pad, int(round((y0 - cy0) * s)) - pad,
                          int(round((x1 - cx0) * s)) + pad, int(round((y1 - cy0) * s)) + pad,
                          int(thickness), self.COLORS[color])
        info = f"кадр {cw}x{ch} @({cx0},{cy0}) -> {nw}x{nh}"
        return (crop[None], info)
