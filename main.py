from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler
import torch
import numpy as np
from PIL import Image

# =================== Downloading models ===================

# ------------------- img2img pipeline -------------------

# ControlNet：Input Segmentation map
# HuggingFace Doc: [https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet]
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
)

# Backbone: img2img generation (passing a text prompt and an initial image to condition the generation of new images)
# HuggingFace Doc: [https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/img2img]
img2img_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
# Use DPM-Solver++ scheduler for faster and more accurate sampling (Still takes about 1 hour)
img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        img2img_pipe.scheduler.config, algorithm_type="dpmsolver++")

# ------------------- inpaint pipeline -------------------
# Backbone: inpainting (edit specific parts of an image by providing a mask and a text prompt)
# HuggingFace Doc: [https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/inpaint]
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")
# Use DPM-Solver++ scheduler for faster and more accurate sampling
inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    inpaint_pipe.scheduler.config, algorithm_type="dpmsolver++"
)
# =========================== End ===========================

# ==================== Utility Functions ====================
# Convert Blender segmentation (RGBA) into:
# - Colored palette mask for ControlNet-Img2Img (binary_mask=False)
# - Binary white-on-black mask for inpainting (binary_mask=True)
def remap_mask_color(seg_path, out_path, binary_mask = False):
    # Load RGBA segmentation exported from Blender
    seg = Image.open(seg_path).convert("RGBA")
    seg_np = np.array(seg)
    # Extract alpha channel: non-zero alpha indicates object regions
    alpha = seg_np[:, :, 3]
    mask = alpha > 0

    # Use ADE20K 150 classes palette
    if binary_mask:
        # The area to inpaint is represented by white pixels and the area to keep is represented by black pixels
        person_color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        # sd-controlnet-seg has strict color‑coding, which uses ADE20K/Coco‐Stuff, so it recognizes that fixed color‑to‑class mapping!
        # ADE20K "person" palette color: magenta-like [150, 5, 61]
        # Color Coding: [https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit?gid=0#gid=0]
        person_color = np.array([150, 5, 61], dtype=np.uint8)

    h, w = mask.shape
    out_np = np.zeros((h, w, 3), dtype=np.uint8)
    out_np[mask] = person_color

    # Print unique colors for verification
    uni = np.unique(out_np.reshape(-1,3), axis=0)
    print(uni)

    Image.fromarray(out_np).save(out_path)

# Perform Img2Img with ControlNet-SEG:
# - Keeps original composition while enhancing details
# - Conditions on segmentation layout
def refine_render(rgb_path, seg_path):

    init_image = Image.open(rgb_path).convert("RGB")
    control_image = Image.open(seg_path).convert("RGB")

    # Crop to nearest multiples of 8 (to satisfy model requirements)
    w, h = init_image.size
    w8, h8 = (w // 8) * 8, (h // 8) * 8

    init_image = init_image.resize((w8, h8), Image.BILINEAR)
    control_image = control_image.resize((w8, h8), Image.NEAREST)

    # Run the pipeline: balance between preservation and enhancement
    out = img2img_pipe(
        prompt=(
            "photorealistic, high resolution, realistic lighting and textures, "
            "keep original composition and layout"
        ),
        negative_prompt="low resolution, artifacts, extra objects",
        image=init_image,
        control_image=control_image,
        # Lower -> keep more init_image details
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=40
    ).images[0]

    out.save(rgb_path.replace(".png", "_refined.png"))

# Perform inpainting on specified mask regions:
# - Only white regions in mask are edited
# - Background preserved entirely
def inpaint_render(rgb_path, seg_path):
    # Load input image and binary mask (single-channel)
    init_image = Image.open(rgb_path).convert("RGB")
    mask = Image.open(seg_path).convert("L")

    # Crop to nearest multiples of 8
    w, h = init_image.size
    w8, h8 = (w//8)*8, (h//8)*8
    init_image = init_image.resize((w8, h8), Image.BILINEAR)
    mask = mask.resize((w8, h8), Image.NEAREST)

    # Run inpainting: white mask areas are repainted
    out = inpaint_pipe(
        prompt=(
            "photorealistic, high resolution, realistic lighting and textures person, "
            "keep original posture"
        ),
        negative_prompt="low resolution, artifacts, extra objects",
        image=init_image,
        mask_image=mask,
        # Control inpaint strength: 0 -> keep the original img, 1 -> Re-generate the whole img
        strength=0.4,
        guidance_scale=7.5,
        num_inference_steps=40
    ).images[0]

    out.save(rgb_path.replace(".png", "_inpaint.png"))

#remap_mask_color("masks/seg.png", "masks/seg_person_palette.png", binary_mask=False)
remap_mask_color("masks/seg.png", "masks/seg_mask_binary.png", binary_mask=True)

#refine_render("images/render.png", "masks/seg_person_palette.png")
inpaint_render("images/render.png", "masks/seg_mask_binary.png")