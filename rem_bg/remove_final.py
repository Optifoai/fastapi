import cv2
import numpy as np
from rembg import remove
from PIL import Image

def remove_bg_car_shadow(image_path, output_path="output.png",
                         feather=3, shadow_gamma=1.5, shadow_darkening=0.75,
                         crop=True, padding=20):
    """
    Remove background but keep only under-car shadows, then crop final PNG
    to car area (optional).
    """

    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Step 1: Background removal
    input_pil = Image.fromarray(image)
    no_bg = remove(input_pil)
    no_bg_np = np.array(no_bg)

    # Step 2: Extract alpha
    if no_bg_np.shape[2] == 4:
        alpha = no_bg_np[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(no_bg_np.shape[:2], dtype=np.float32)

    # Step 3: Feather slightly
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)

    # Step 4 & 5: Identify car body and shadow zone
    car_mask = (alpha > 0.9).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    shadow_zone_mask = cv2.dilate(car_mask, kernel, iterations=2)

    # Step 6: Boost alpha inside shadow zone
    boosted_alpha = alpha.copy()
    inside_shadow_zone = shadow_zone_mask > 0
    boosted_alpha[inside_shadow_zone] = np.power(alpha[inside_shadow_zone], 1.0 / shadow_gamma)
    boosted_alpha = np.clip(boosted_alpha, 0, 1)

    # Step 7: Create RGBA
    result = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (boosted_alpha * 255).astype(np.uint8)

    # Step 8: Darken RGB in semi-shadow areas
    semi_shadow = (boosted_alpha > 0.1) & (boosted_alpha < 0.9) & inside_shadow_zone
    result[semi_shadow, 0:3] = (result[semi_shadow, 0:3] * shadow_darkening).astype(np.uint8)

    # Step 9 (NEW): Crop to car+shadow bounding box
    if crop:
        alpha_mask = result[:, :, 3]
        coords = cv2.findNonZero((alpha_mask > 10).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Apply padding safely
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, result.shape[1] - x)
            h = min(h + 2 * padding, result.shape[0] - y)

            result = result[y:y+h, x:x+w]

    cv2.imwrite(output_path, result)
    print(f"✅ Saved refined cropped car+shadow cutout → {output_path}")


# Example usage
remove_bg_car_shadow(
    "car_with_shadow.jpg", "car_shadow_refined_cropped.png",
    feather=3, shadow_gamma=1.6, shadow_darkening=0.7,
    crop=True, padding=30
)
