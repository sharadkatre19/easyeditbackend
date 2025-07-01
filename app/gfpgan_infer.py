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

def restore_image(input_path: str, output_path: str, brightness_factor: float = 1.15):
    """
    Restore face from an image and save to output path with brightness adjustment
    
    Args:
        input_path: Path to input image
        output_path: Path to save restored image
        brightness_factor: Factor to increase brightness (1.0 = no change, 1.3 = 30% brighter)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {input_path}")

    # Convert to RGB (GFPGAN expects RGB input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        _, _, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # Convert back to BGR for OpenCV processing
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        # --- Brightness Enhancement ---
        # Method 1: Simple brightness multiplication (recommended)
        brightened_img = cv2.multiply(restored_img, brightness_factor)
        
        # Ensure values don't exceed 255
        brightened_img = np.clip(brightened_img, 0, 255).astype(np.uint8)
        
        # Alternative Method 2: HSV brightness adjustment (uncomment to use instead)
        # hsv = cv2.cvtColor(restored_img, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v = cv2.multiply(v, brightness_factor)
        # v = np.clip(v, 0, 255).astype(np.uint8)
        # brightened_img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        
        # Alternative Method 3: Gamma correction (uncomment to use instead)
        # gamma = 0.7  # Lower gamma = brighter image (try values between 0.5-0.8)
        # lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # brightened_img = cv2.LUT(restored_img, lookup_table)

        # --- Color Correction (optional tweak) ---
        # Convert to HSV and adjust saturation if needed
        hsv = cv2.cvtColor(brightened_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Slightly reduce saturation to make color more natural
        s = cv2.multiply(s, 0.95)  # Reduce saturation by 5%

        hsv_corrected = cv2.merge([h, s.astype(np.uint8), v])
        final_img = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

        # Save restored and brightened image
        cv2.imwrite(output_path, final_img)
        
    except Exception as e:
        raise RuntimeError(f"Image restoration failed: {e}")

# Alternative function with more granular control
def restore_image_advanced(input_path: str, output_path: str, 
                          brightness: int = 30, contrast: float = 1.1):
    """
    Restore face with advanced brightness and contrast control
    
    Args:
        input_path: Path to input image
        output_path: Path to save restored image  
        brightness: Brightness adjustment (-100 to 100, positive = brighter)
        contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {input_path}")

    # Convert to RGB (GFPGAN expects RGB input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        _, _, restored_img = restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # Convert back to BGR
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

        # Apply brightness and contrast
        # Formula: new_img = contrast * img + brightness
        adjusted_img = cv2.convertScaleAbs(restored_img, alpha=contrast, beta=brightness)

        # Color correction
        hsv = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 0.95)
        hsv_corrected = cv2.merge([h, s.astype(np.uint8), v])
        final_img = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

        cv2.imwrite(output_path, final_img)
        
    except Exception as e:
        raise RuntimeError(f"Image restoration failed: {e}")

# Example usage:
# restore_image("input.jpg", "output.jpg", brightness_factor=1.4)  # 40% brighter
# restore_image_advanced("input.jpg", "output.jpg", brightness=40, contrast=1.2)