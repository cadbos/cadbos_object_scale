from .common import _resize_mask, _bbox


class MaskAmodalExtend:
    """Если низ объекта закрыт предметом перед ним (стол, диван), достраивает
    рамку маски вниз до полной высоты по пропорциям нового объекта.
    Тогда новый объект вписывается от верхнего края видимой части
    и не «висит» над преградой."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Видимая часть объекта."}),
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "reference_mask": ("MASK", {"tooltip": "Маска нового объекта (выход mask узла 158): "
                                                   "по ней берутся пропорции."}),
            "band": ("INT", {"default": 12, "min": 2, "max": 128,
                             "tooltip": "Высота полосы под объектом, px, где ищется преграда."}),
            "margin": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.5, "step": 0.005,
                                 "tooltip": "Насколько полоса должна быть ближе к камере, "
                                            "чем низ объекта, чтобы считаться преградой."}),
            "min_occluded": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "Какая доля полосы должна быть закрыта."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "occluders": ("MASK", {"tooltip": "Другие найденные предметы (узел 258). Если "
                                              "подключено, преградой считаются только они, "
                                              "а не пол."}),
        }}

    def run(self, mask, depth, reference_mask, band, margin, min_occluded,
            depth_near_is_bright, occluders=None):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        dev = m.device
        obj = (m > 0.5).float()
        bb = _bbox(obj)
        if bb is None:
            return (obj[None], "маска пустая")
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        if y1 >= H - 2:
            return (obj[None], "объект доходит до края кадра — без изменений")

        d = depth[0].float().to(dev)
        if d.dim() == 3:
            d = d.mean(dim=-1)
        d = _resize_mask(d, H, W)
        if not depth_near_is_bright:
            d = 1.0 - d
        foot_rows = obj[max(y0, y1 - max(2, int(0.1 * h))):y1]
        foot_vals = d[max(y0, y1 - max(2, int(0.1 * h))):y1][foot_rows > 0.5]
        if foot_vals.numel() == 0:
            return (obj[None], "не удалось оценить глубину низа объекта")
        d_foot = float(foot_vals.median())
        strip = d[y1:min(H, y1 + int(band)), x0:x1]
        near = strip > d_foot + float(margin)
        if occluders is not None:   # преграда — только найденный предмет, не пол
            o = occluders.float().to(dev)
            if o.dim() == 3:
                o = o.amax(dim=0)
            o = _resize_mask(o, H, W) > 0.5
            near = near & o[y1:min(H, y1 + int(band)), x0:x1]
        occ = float(near.float().mean())
        if occ < float(min_occluded):
            return (obj[None], f"низ объекта виден (закрыто {100 * occ:.0f}% полосы) — "
                               f"без изменений")

        r = reference_mask.float().to(dev)
        if r.dim() == 3:
            r = r.amax(dim=0)
        rb = _bbox(r)
        if rb is None:
            return (obj[None], "маска нового объекта пустая — без изменений")
        aspect = (rb[3] - rb[1]) / max(1, rb[2] - rb[0])
        h_full = int(round(w * aspect))
        if h_full <= h:
            return (obj[None], f"низ закрыт ({100 * occ:.0f}%), но полная высота "
                               f"{h_full}px не больше видимой {h}px — без изменений")
        out = obj.clone()
        out[y1:min(H, y0 + h_full), x0:x1] = 1.0
        return (out[None], f"низ закрыт преградой ({100 * occ:.0f}% полосы): рамка "
                           f"достроена вниз с {h} до {min(H, y0 + h_full) - y0}px")
