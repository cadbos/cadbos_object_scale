import torch
import torch.nn.functional as F

from .common import _resize_mask, _bbox, _push_pull_fill


class ObjectEdgePad:
    """Если объект обрезан краем кадра, расширяет кадр с этих сторон, чтобы
    объект поместился целиком. Новая область заливается мягким цветом соседних
    пикселей, pad_mask — что нужно дорисовать."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "pad_mask", "left", "top", "right", "bottom", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Исходная сцена (узел 1)."}),
            "mask": ("MASK", {"tooltip": "Старый объект, найденный на уменьшенной копии."}),
            "amount": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "На сколько расширять кадр: доля видимого размера объекта."}),
            "max_fraction": ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Не больше этой доли размера кадра с каждой стороны."}),
            "edge_px": ("INT", {
                "default": 2, "min": 0, "max": 64,
                "tooltip": "Объект считается обрезанным, если ближе к краю, px."}),
            "multiple": ("INT", {"default": 16, "min": 1, "max": 128}),
            "keep_aspect": ("BOOLEAN", {
                "default": True,
                "tooltip": "Сохранять пропорции кадра: досыпать отступ по другой оси, "
                        "чтобы область вставки не выходила за край."}),
        }}

    def run(self, image, mask, amount, max_fraction, edge_px, multiple,
            keep_aspect=True):
        img = image[0].float()[..., :3]
        H, W, _ = img.shape
        dev = img.device
        m = mask.float().to(dev)
        if m.dim() == 3:
            m = m.amax(dim=0)
        m = (_resize_mask(m, H, W) > 0.5).float()
        z = torch.zeros((1, H, W), device=dev)
        bb = _bbox(m)
        if bb is None:
            return (image, z, 0, 0, 0, 0, "объект не найден — кадр без изменений")
        x0, y0, x1, y1 = bb
        ow, oh = x1 - x0, y1 - y0

        def pad(size, frame):
            p = min(float(amount) * size, float(max_fraction) * frame)
            return max(0, int(round(p / multiple)) * multiple)

        left = pad(ow, W) if x0 <= edge_px else 0
        right = pad(ow, W) if x1 >= W - edge_px else 0
        top = pad(oh, H) if y0 <= edge_px else 0
        bottom = pad(oh, H) if y1 >= H - edge_px else 0
        if keep_aspect and left + right + top + bottom > 0:
            def up(v):
                return int(-(-v // multiple) * multiple)  # округление вверх до кратного
            NW, NH = W + left + right, H + top + bottom
            need_h = up(max(0, round(NW * H / W) - NH))
            need_w = up(max(0, round(NH * W / H) - NW))
            if need_h > 0:
                if bottom and not top:
                    bottom += need_h
                elif top and not bottom:
                    top += need_h
                else:
                    top += need_h // 2
                    bottom += need_h - need_h // 2
            if need_w > 0:
                if right and not left:
                    right += need_w
                elif left and not right:
                    left += need_w
                else:
                    left += need_w // 2
                    right += need_w - need_w // 2
        if left + right + top + bottom == 0:
            return (image, z, 0, 0, 0, 0, "объект не касается края — кадр без изменений")

        NH, NW = H + top + bottom, W + left + right
        out = torch.zeros((NH, NW, 3), device=dev)
        out[top:top + H, left:left + W] = img
        pm = torch.ones((NH, NW), device=dev)
        pm[top:top + H, left:left + W] = 0
        out = _push_pull_fill(out, pm)
        info = (f"кадр расширен: слева {left}, сверху {top}, справа {right}, снизу {bottom} px; "
                f"{W}x{H} -> {NW}x{NH}")
        return (out[None].clamp(0, 1), pm[None], left, top, right, bottom, info)


class OutpaintSwitch:
    """Возвращает дорисованный кадр, только если кадр расширялся. Иначе — исходный,
    и ветка дорисовки не выполняется (экономия времени)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Выход image узла Object Edge Pad."}),
            "pad_mask": ("MASK", {"tooltip": "Выход pad_mask узла Object Edge Pad."}),
            "outpainted": ("IMAGE", {"lazy": True, "tooltip": "Результат дорисовки."}),
        }}

    def check_lazy_status(self, image=None, pad_mask=None, outpainted=None):
        if pad_mask is not None and float(pad_mask.max()) > 0.5 and outpainted is None:
            return ["outpainted"]
        return []

    def run(self, image, pad_mask, outpainted=None):
        if float(pad_mask.max()) > 0.5 and outpainted is not None:
            return (outpainted,)
        return (image,)


class ObjectEdgeCrop:
    """Обрезает результат до исходного кадра и накладывает его на исходную
    картинку только там, где были изменения (остальное — без потерь)."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Результат в размере расширенного кадра (узел 67)."}),
            "mask": ("MASK", {"tooltip": "Где были изменения (выход mask узла 67)."}),
            "original": ("IMAGE", {"tooltip": "Исходная сцена (узел 1)."}),
            "left": ("INT", {"forceInput": True}),
            "top": ("INT", {"forceInput": True}),
            "right": ("INT", {"forceInput": True}),
            "bottom": ("INT", {"forceInput": True}),
        }}

    def run(self, image, mask, original, left, top, right, bottom):
        org = original[0].float()[..., :3]
        H, W, _ = org.shape
        dev = org.device
        res = image[0].float().to(dev)[..., :3]
        m = mask.float().to(dev)
        if m.dim() == 3:
            m = m[0]
        NH, NW = H + top + bottom, W + left + right
        if res.shape[0] != NH or res.shape[1] != NW:
            res = F.interpolate(res.permute(2, 0, 1)[None], size=(NH, NW), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0)
        if m.shape[0] != NH or m.shape[1] != NW:
            m = _resize_mask(m, NH, NW)
        rc = res[top:top + H, left:left + W]
        mc = m[top:top + H, left:left + W].clamp(0, 1)[..., None]
        out = org * (1 - mc) + rc * mc
        return (out[None].clamp(0, 1),)
