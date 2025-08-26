"""Centralized constants for model defaults and schedules.

Follow the paper's defaults; update values here to change across code paths.
"""

# RGB noise schedule defaults
SIGMA_RGB0_DEFAULT: float = 64.0
# For multimodal JetFormer path (text-to-image), we end at 3.0 per paper
SIGMA_RGB_FINAL_MULTIMODAL: float = 3.0
# For class-conditional ImageNet flows, we end at 0.0
SIGMA_RGB_FINAL_IMAGENET: float = 0.0


