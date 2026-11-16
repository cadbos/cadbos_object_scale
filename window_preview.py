import torch

from .common import _resize_mask, _bbox, _draw_rect


class WindowPreview:
    """Предпросмотр для отладки: рамка окна (зелёная), выбранный объект
    (красный) и остальные найденные предметы (синие) поверх сцены."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Сцена (узел 3)."}),
            "window": ("MASK", {"tooltip": "Выход window узла Mask Window."}),
            "thickness": ("INT", {"default": 4, "min": 1, "max": 32}),
        },
        "optional": {
            "selected": ("MASK", {"tooltip": "Выбранный объект (выход mask узла 400)."}),
            "others": ("MASK", {"tooltip": "Остальные предметы (выход others узла 400)."}),
        }}

    def run(self, image, window, thickness, selected=None, others=None):
        img = image[0].float()[..., :3].clone()
        H, W, _ = img.shape
        dev = img.device

        def prep(m):
            m = m.float().to(dev)
            if m.dim() == 3:
                m = m.amax(dim=0)
            return _resize_mask(m, H, W).clamp(0, 1)

        for m, color, alpha in ((others, (0.0, 0.4, 1.0), 0.35),
                                (selected, (1.0, 0.0, 0.0), 0.45)):
            if m is not None:
                a = (prep(m) * alpha)[..., None]
                img = img * (1 - a) + torch.tensor(color, device=dev) * a
        bb = _bbox(prep(window))
        if bb is not None:
            img = _draw_rect(img.contiguous(), bb[0], bb[1], bb[2], bb[3],
                             int(thickness), (0.0, 1.0, 0.0))
        return (img[None].clamp(0, 1),)
