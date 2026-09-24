# Comfyui-DHan-Composite v0.2.0

A lightweight, model-agnostic ComfyUI compositor for blending generated image edits back onto a source image while preserving untouched regions.

Comfyui-DHan-Composite is designed as a post-processing step for image-editing workflows. The edited image can come from an API model such as Nano Banana / Gemini or GPT Image, a local image-edit model, or any other workflow that outputs a standard ComfyUI `IMAGE`.

The node does not depend on the model that produced the edit. It works from the source image, generated image, and edit mask, then builds a composite mask, feathers the transition, and optionally performs simple seam color matching.

## Typical workflow

```text
Source Image + Edit Mask
          |
          v
Image-edit model / API
(Nano Banana, GPT Image, local model, etc.)
          |
          v
    Comfyui-DHan-Composite
          |
          v
      Final Image
```

## Inputs

- `generated_image` — edited/generated image returned by an API or local model
- `source_image` — original source/canvas image to preserve outside the composite region
- `edit_mask` — authored edit mask
- `mask_mode` — ARC Mask / ARC Mask + Detected Changes / Detected Changes
- `change_sensitivity` — threshold used when detecting differences between source and generated images
- `mask_grow_px` — expands the working composite mask
- `edge_feather_px` — softens the transition between source and generated regions
- `seam_match` — applies lightweight color-shift matching near the composite seam
- `debug` — outputs a diagnostic comparison view

## Outputs

- `image` — final composite
- `composite_mask` — mask used for the composite
- `report` — text summary of the composite operation
- `debug_view` — optional source/generated/mask/result diagnostic view

## Notes

- The generated image is automatically resized to the source image dimensions when needed.
- Best results come from generated edits that remain spatially aligned with the source image.
- Difference detection can be used by itself or combined with the authored edit mask.
- Pure PyTorch implementation with no OpenCV, SciPy, or scikit-image runtime dependency.

## Compatibility

Comfyui-DHan-Composite operates on standard ComfyUI `IMAGE` and `MASK` types, so it is not tied to a specific image-generation model or API.

Examples include:

- Nano Banana / Gemini image editing
- GPT Image editing
- Local image-edit models
- Other API or custom workflows that return a ComfyUI image

## License

GPL-3.0

## Author

Developed by DHan315.
