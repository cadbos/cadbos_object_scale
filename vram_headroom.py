import comfy.model_management as mm


class VRAMHeadroom:
    """Пропускает картинку без изменений, но перед этим освобождает видеопамять:
    выгружает модели ComfyUI (Qwen, текстовые энкодеры), пока свободно меньше
    free_gb. Ставится перед узлами, которые грузят модели мимо ComfyUI
    (GroundingDINO+SAM, RMBG, Depth Anything), — ComfyUI о них не знает
    и сам память под них не освобождает."""
    CATEGORY = "cadbos/utils"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "free_gb": ("FLOAT", {
                "default": 4.0, "min": 0.0, "max": 24.0, "step": 0.5,
                "tooltip": "Сколько ГБ видеопамяти должно быть свободно после узла."}),
        }}

    def run(self, image, free_gb):
        dev = mm.get_torch_device()
        before = mm.get_free_memory(dev) / 1024 ** 3
        need = float(free_gb) * 1024 ** 3
        if before * 1024 ** 3 < need:
            mm.free_memory(need, dev)
            mm.soft_empty_cache()
        after = mm.get_free_memory(dev) / 1024 ** 3
        return (image, f"свободно VRAM: {before:.1f} -> {after:.1f} ГБ")
