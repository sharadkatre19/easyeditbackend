import cv2
import numpy as np
from gfpgan import GFPGANer

# Initialize GFPGAN once at module level
model_path = 'weights/GFPGANv1.4.pth'
restorer = GFPGANer(
    model_path=model_path,
    upscale=4,  # You can increase this to 4 for higher resolution if needed
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def restore_image(input_path: str, output_path: str):
    """Restore face from an image and save to output path"""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {input_path}")

    # Convert to RGB (GFPGAN expects RGB input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        _, _, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # Optionally: adjust color balance to reduce reddish hue
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        # --- Color Correction (optional tweak) ---
        # Convert to HSV and reduce red channel if needed
        hsv = cv2.cvtColor(restored_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Slightly reduce saturation to make color more natural
        s = cv2.multiply(s, 0.8)

        hsv_corrected = cv2.merge([h, s.astype(np.uint8), v])
        restored_img = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

        # Save restored image
        cv2.imwrite(output_path, restored_img)
    except Exception as e:
        raise RuntimeError(f"Image restoration failed: {e}")
