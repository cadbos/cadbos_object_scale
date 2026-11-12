import torch
import numpy as np
import scipy.ndimage as ndi

from .common import _resize_mask


def _split(a, separate, dep, depth_gap):
    """Делит маску на отдельные предметы. Касающиеся предметы разделяются
    эрозией на separate px; каждый пиксель возвращается ближайшему «ядру»
    внутри своего связного куска."""
    orig, n0 = ndi.label(a)
    lab = np.zeros(a.shape, dtype=np.int32)
    n = 0
    for sl_i, sl in enumerate(ndi.find_objects(orig), start=1):
        c = orig[sl] == sl_i
        core = ndi.binary_erosion(c, iterations=int(separate)) if separate > 0 else c
        cl, cn = ndi.label(core)
        if cn <= 1:
            n += 1
            lab[sl][c] = n
            continue
        _, (iy, ix) = ndi.distance_transform_edt(cl == 0, return_indices=True)
        part = cl[iy, ix]
        lab[sl][c] = part[c] + n
        n += cn
    # куски одного предмета, разрезанного другим (спинка над столом и ножки
    # под ним): по горизонтали друг над другом и на одной глубине — склеить
    if dep is not None and n > 1:
        info = {}
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if len(xs):
                info[i] = (xs.min(), xs.max(), float(np.median(dep[ys, xs])))
        ids = sorted(info)
        for a_i in ids:
            for b_i in ids:
                if b_i <= a_i or a_i not in info or b_i not in info:
                    continue
                ax0, ax1, ad = info[a_i]
                bx0, bx1, bd = info[b_i]
                ov = min(ax1, bx1) - max(ax0, bx0)
                narrow = min(ax1 - ax0, bx1 - bx0) + 1
                if ov > 0.6 * narrow and abs(ad - bd) < depth_gap:
                    lab[lab == b_i] = a_i
                    ys, xs = np.nonzero(lab == a_i)
                    info[a_i] = (xs.min(), xs.max(), float(np.median(dep[ys, xs])))
                    del info[b_i]
    return lab


def _pick(a, mode, index, px, py, separate, min_fraction, dep, ref, depth_gap=0.05):
    H, W = a.shape
    lab = _split(a, separate, dep, depth_gap)
    ids = [int(i) for i in np.unique(lab) if i > 0]
    st = []
    for i in ids:
        c = lab == i
        ys, xs = np.nonzero(c)
        st.append(dict(id=i, area=len(xs), cx=float(xs.mean()), bottom=int(ys.max()),
                       depth=float(np.median(dep[c])) if dep is not None else None,
                       ov=int((c & ref).sum()) if ref is not None else 0))
    big = max(s["area"] for s in st)
    st = [s for s in st if s["area"] >= min_fraction * big]
    st.sort(key=lambda s: s["cx"])  # нумерация слева направо
    if mode == "largest":
        k = max(range(len(st)), key=lambda j: st[j]["area"])
    elif mode == "nearest":
        if dep is not None:
            k = max(range(len(st)), key=lambda j: st[j]["depth"])
        else:  # без глубины: чей низ ниже в кадре, тот ближе
            k = max(range(len(st)), key=lambda j: st[j]["bottom"])
    elif mode == "leftmost":
        k = 0
    elif mode == "rightmost":
        k = len(st) - 1
    elif mode == "center":
        k = min(range(len(st)), key=lambda j: abs(st[j]["cx"] - W / 2))
    elif mode == "index":
        k = max(0, min(len(st) - 1, int(index)))
    elif mode == "point":
        x, y = px * (W - 1), py * (H - 1)
        def dist(s):
            ys, xs = np.nonzero(lab == s["id"])
            return float(((xs - x) ** 2 + (ys - y) ** 2).min())
        k = min(range(len(st)), key=lambda j: dist(st[j]))
    else:  # overlap
        if ref is not None and max(s["ov"] for s in st) > 0:
            k = max(range(len(st)), key=lambda j: st[j]["ov"])
        else:
            k = max(range(len(st)), key=lambda j: st[j]["area"])
    sel = lab == st[k]["id"]
    lines = []
    for j, s in enumerate(st):
        d = "" if s["depth"] is None else f", глубина {s['depth']:.2f}"
        mark = "  <- выбран" if j == k else ""
        lines.append(f"#{j}: центр x={100 * s['cx'] / W:.0f}%, площадь {s['area']}{d}{mark}")
    info = f"найдено экземпляров: {len(st)} (режим {mode})\n" + "\n".join(lines)
    return sel, a & ~sel, info


class MaskPickInstance:
    """Если по названию найдено несколько одинаковых предметов (например,
    несколько стульев), оставляет из маски только один. others — остальные."""
    CATEGORY = "cadbos/object"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("mask", "others", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK", {"tooltip": "Все найденные предметы (GroundingDINO + SAM)."}),
            "mode": (["nearest", "largest", "leftmost", "rightmost", "center",
                      "index", "point", "overlap"], {
                "default": "nearest",
                "tooltip": "nearest — ближайший к камере; largest — самый крупный; "
                           "leftmost / rightmost / center — по положению в кадре; "
                           "index — по номеру слева направо (см. info); "
                           "point — ближайший к точке point_x/point_y; "
                           "overlap — сильнее всего пересекается с reference."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 64}),
            "point_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Доля ширины кадра (0 — левый край)."}),
            "point_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Доля высоты кадра (0 — верх)."}),
            "separate": ("INT", {"default": 6, "min": 0, "max": 64,
                                 "tooltip": "Разделять предметы, соединённые перемычкой "
                                            "уже ~2 x значение px."}),
            "min_fraction": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Куски меньше этой доли самого крупного — мусор."}),
        },
        "optional": {
            "depth": ("IMAGE", {"tooltip": "Карта глубины (узел 105)."}),
            "reference": ("MASK", {"tooltip": "Для режима overlap."}),
            "depth_near_is_bright": ("BOOLEAN", {"default": True}),
        }}

    def run(self, mask, mode, index, point_x, point_y, separate, min_fraction,
            depth=None, reference=None, depth_near_is_bright=True):
        m = mask.float()
        if m.dim() == 3:
            m = m.amax(dim=0)
        H, W = m.shape
        a = (m > 0.5).cpu().numpy()
        to_t = lambda x: torch.from_numpy(x.astype(np.float32))[None].to(mask.device)
        if not a.any():
            return (to_t(a), to_t(a), "маска пустая")
        dep = None
        if depth is not None:
            d = depth[0].float()
            if d.dim() == 3:
                d = d.mean(dim=-1)
            d = _resize_mask(d.to(m.device), H, W)
            if not depth_near_is_bright:
                d = 1.0 - d
            dep = d.cpu().numpy()
        ref = None
        if reference is not None:
            r = reference.float().to(m.device)
            if r.dim() == 3:
                r = r.amax(dim=0)
            ref = (_resize_mask(r, H, W) > 0.5).cpu().numpy()
        sel, others, info = _pick(a, mode, index, float(point_x), float(point_y),
                                  int(separate), float(min_fraction), dep, ref)
        return (to_t(sel), to_t(others), info)
