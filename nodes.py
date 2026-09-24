"""Comfyui-DHan-Composite v0.2.0

Lightweight, model-agnostic post-generation compositor for ComfyUI workflows.
Designed to preserve source regions while blending generated image edits.
Pure PyTorch; no OpenCV/scipy/skimage runtime dependency.
"""

import torch
import torch.nn.functional as F

_VERSION = "0.2.0"


def _image4(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Comfyui-DHan-Composite expected IMAGE [B,H,W,C], got {tuple(x.shape)}")
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 1, 3)
    elif x.shape[-1] == 4:
        x = x[..., :3]
    elif x.shape[-1] != 3:
        raise ValueError(f"Comfyui-DHan-Composite expected RGB/RGBA image, got {tuple(x.shape)}")
    return x.clamp(0.0, 1.0)


def _resize_image(x, h, w):
    if x.shape[1:3] == (h, w):
        return x
    x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _mask3(x, batch, h, w, device):
    if x is None:
        return torch.zeros((batch, h, w), dtype=torch.float32, device=device)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.detach().float().to(device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 3:
        raise ValueError(f"Comfyui-DHan-Composite expected MASK [B,H,W], got {tuple(x.shape)}")
    if x.shape[0] == 1 and batch > 1:
        x = x.repeat(batch, 1, 1)
    elif x.shape[0] != batch:
        x = x[:1].repeat(batch, 1, 1)
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return x.clamp(0.0, 1.0)


def _dilate(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    return F.max_pool2d(mask.unsqueeze(1), 2 * radius + 1, 1, radius)[:, 0]


def _erode(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    return -F.max_pool2d((-mask).unsqueeze(1), 2 * radius + 1, 1, radius)[:, 0]


def _feather(mask, radius):
    radius = max(0, int(radius))
    if radius == 0:
        return mask.clamp(0.0, 1.0)
    k = 2 * radius + 1
    x = mask.unsqueeze(1)
    # Two box passes produce a smooth, dependency-free edge rolloff.
    x = F.avg_pool2d(x, k, 1, radius)
    x = F.avg_pool2d(x, k, 1, radius)
    return x[:, 0].clamp(0.0, 1.0)


def _change_mask(source, generated, sensitivity):
    diff = torch.sqrt(torch.sum((generated - source) ** 2, dim=-1)).clamp_min(0.0)
    # 0.0 = permissive, 1.0 = only strong changes. Keep a small floor for JPEG/API noise.
    threshold = 0.025 + (float(sensitivity) * 0.20)
    return (diff > threshold).float(), threshold


def _seam_match(source, generated, mask, strength, width):
    strength = float(strength)
    if strength <= 0.0:
        return generated
    hard = (mask > 0.5).float()
    inner = _erode(hard, max(1, int(width)))
    ring = (hard - inner).clamp(0.0, 1.0)
    ring3 = ring.unsqueeze(-1)
    denom = ring3.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    shift = ((source - generated) * ring3).sum(dim=(1, 2), keepdim=True) / denom
    corrected = (generated + shift).clamp(0.0, 1.0)
    weight = (ring3 * strength).clamp(0.0, 1.0)
    return generated * (1.0 - weight) + corrected * weight


class ARCCompositeNode:
    """Model-agnostic compositor for image-editing workflows."""

    @classmethod
    def VALIDATE_INPUTS(cls, mask_mode):
        if mask_mode in ("Edit Mask", "Edit Mask + Detected Changes", "Detected Changes",
                         "ARC Mask", "ARC Mask + Detected Changes"):
            return True
        return f"Unknown mask mode: {mask_mode}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "edit_mask": ("MASK",),
                "mask_mode": (["Edit Mask", "Edit Mask + Detected Changes", "Detected Changes"], {"default": "Edit Mask"}),
                "change_sensitivity": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_grow_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "edge_feather_px": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "seam_match": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "composite_mask", "report", "debug_view")
    FUNCTION = "run"
    CATEGORY = "Comfyui-DHan/Composite"
    DESCRIPTION = "Model-agnostic masked compositor for blending API or local image edits back onto a source image."

    def run(self, generated_image, source_image, edit_mask, mask_mode,
            change_sensitivity, mask_grow_px, edge_feather_px, seam_match, debug):
        source = _image4(source_image)
        generated = _image4(generated_image).to(source.device)

        batch = max(source.shape[0], generated.shape[0])
        if source.shape[0] == 1 and batch > 1:
            source = source.repeat(batch, 1, 1, 1)
        if generated.shape[0] == 1 and batch > 1:
            generated = generated.repeat(batch, 1, 1, 1)
        if source.shape[0] != batch:
            source = source[:1].repeat(batch, 1, 1, 1)
        if generated.shape[0] != batch:
            generated = generated[:1].repeat(batch, 1, 1, 1)

        h, w = source.shape[1:3]
        generated = _resize_image(generated, h, w)
        authored = (_mask3(edit_mask, batch, h, w, source.device) > 0.5).float()
        detected, threshold = _change_mask(source, generated, change_sensitivity)

        if mask_mode == "Detected Changes":
            working = detected
        elif mask_mode in ("Edit Mask + Detected Changes", "ARC Mask + Detected Changes"):
            working = torch.maximum(authored, detected)
        else:
            working = authored

        if int(mask_grow_px) > 0:
            working = _dilate(working, int(mask_grow_px))

        hard_mask = (working > 0.5).float()
        matched = _seam_match(source, generated, hard_mask, seam_match, max(2, int(edge_feather_px) // 2))
        alpha = _feather(hard_mask, int(edge_feather_px)).unsqueeze(-1)
        result = (source * (1.0 - alpha) + matched * alpha).clamp(0.0, 1.0)

        final_px = int(hard_mask.sum().item())
        total_px = batch * h * w
        report = "\n".join([
            "COMFYUI-DHAN COMPOSITE",
            f"Version: {_VERSION}",
            f"Canvas: {w}x{h}",
            f"Mask mode: {mask_mode.replace('ARC Mask', 'Edit Mask')}",
            f"Detected-change threshold: {threshold:.4f}",
            f"Mask grow: {int(mask_grow_px)} px",
            f"Edge feather: {int(edge_feather_px)} px",
            f"Seam match: {float(seam_match):.2f}",
            f"Composite area: {final_px}/{total_px} pixels",
        ])

        if debug:
            mask_rgb = hard_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
            top = torch.cat([source, generated], dim=2)
            bottom = torch.cat([mask_rgb, result], dim=2)
            debug_view = torch.cat([top, bottom], dim=1)
        else:
            debug_view = torch.zeros((batch, 1, 1, 3), dtype=result.dtype, device=result.device)

        return result.cpu(), hard_mask.cpu(), report, debug_view.cpu()


NODE_CLASS_MAPPINGS = {
    "ARCCompositeNode": ARCCompositeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ARCCompositeNode": "DHan-Composite",
}

