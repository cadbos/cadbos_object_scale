from .object_scale_placer import ObjectScalePlacer
from .mask_keep_in_front import MaskKeepInFront
from .reference_scale_for_crop import ReferenceScaleForCrop
from .fill_items_on_object import FillItemsOnObject
from .mask_convex_hull import MaskConvexHull
from .mask_keep_outside import MaskKeepOutside
from .exclude_word_from_list import ExcludeWordFromList
from .mask_subtract_parts import MaskSubtractParts
from .mask_fallback import MaskFallback
from .string_fallback import StringFallback
from .object_orientation_text import ObjectOrientationText
from .mask_recover_thin import MaskRecoverThin
from .mask_fill_object_holes import MaskFillObjectHoles
from .mask_clear_top import MaskClearTop
from .object_edge_pad import ObjectEdgePad, OutpaintSwitch, ObjectEdgeCrop
from .mask_pick_instance import MaskPickInstance
from .object_clay_view import ObjectClayView


NODE_CLASS_MAPPINGS = {
    "ObjectScalePlacer": ObjectScalePlacer,
    "MaskKeepInFront": MaskKeepInFront,
    "ReferenceScaleForCrop": ReferenceScaleForCrop,
    "FillItemsOnObject": FillItemsOnObject,
    "MaskConvexHull": MaskConvexHull,
    "MaskKeepOutside": MaskKeepOutside,
    "ExcludeWordFromList": ExcludeWordFromList,
    "MaskSubtractParts": MaskSubtractParts,
    "MaskFallback": MaskFallback,
    "StringFallback": StringFallback,
    "ObjectOrientationText": ObjectOrientationText,
    "MaskRecoverThin": MaskRecoverThin,
    "MaskFillObjectHoles": MaskFillObjectHoles,
    "MaskClearTop": MaskClearTop,
    "ObjectEdgePad": ObjectEdgePad,
    "OutpaintSwitch": OutpaintSwitch,
    "ObjectEdgeCrop": ObjectEdgeCrop,
    "MaskPickInstance": MaskPickInstance,
    "ObjectClayView": ObjectClayView,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ObjectScalePlacer": "Object Scale Placer (cadbos)",
    "MaskKeepInFront": "Mask Keep In Front (depth)",
    "ReferenceScaleForCrop": "Reference Scale For Crop",
    "FillItemsOnObject": "Fill Items On Object",
    "MaskConvexHull": "Mask Convex Hull",
    "MaskKeepOutside": "Mask Keep Outside",
    "ExcludeWordFromList": "Exclude Word From List",
    "MaskSubtractParts": "Mask Subtract Parts",
    "MaskFallback": "Mask Fallback",
    "StringFallback": "String Fallback",
    "ObjectOrientationText": "Object Orientation Text",
    "MaskRecoverThin": "Mask Recover Thin",
    "MaskFillObjectHoles": "Mask Fill Object Holes",
    "MaskClearTop": "Mask Clear Top",
    "ObjectEdgePad": "Object Edge Pad",
    "OutpaintSwitch": "Outpaint Switch",
    "ObjectEdgeCrop": "Object Edge Crop",
    "MaskPickInstance": "Mask Pick Instance",
    "ObjectClayView": "Object Clay View",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
