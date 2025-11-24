import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import csv


def remove_bg_car_shadow(image_path, output_path,
                         feather=3, shadow_gamma=1.5, shadow_darkening=0.75,
                         crop=True, padding=20):
    """Remove background but keep car shadow and crop tightly."""

    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        return False, "Failed to read image"

    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(image)
    no_bg = remove(input_pil)
    no_bg_np = np.array(no_bg)

    if no_bg_np.shape[2] == 4:
        alpha = no_bg_np[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(no_bg_np.shape[:2], dtype=np.float32)

    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)

    car_mask = (alpha > 0.9).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
    shadow_zone_mask = cv2.dilate(car_mask, kernel, iterations=2)

    boosted_alpha = alpha.copy()
    inside_shadow_zone = shadow_zone_mask > 0
    boosted_alpha[inside_shadow_zone] = np.power(alpha[inside_shadow_zone], 1.0 / shadow_gamma)
    boosted_alpha = np.clip(boosted_alpha, 0, 1)

    result = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (boosted_alpha * 255).astype(np.uint8)

    semi_shadow = (boosted_alpha > 0.1) & (boosted_alpha < 0.9) & inside_shadow_zone
    result[semi_shadow, 0:3] = (result[semi_shadow, 0:3] * shadow_darkening).astype(np.uint8)

    if crop:
        alpha_mask = result[:, :, 3]
        coords = cv2.findNonZero((alpha_mask > 10).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + padding * 2, result.shape[1] - x)
            h = min(h + padding * 2, result.shape[0] - y)
            result = result[y:y+h, x:x+w]

    cv2.imwrite(output_path, result)
    return True, "Success"


def process_file(args):
    """Wrapper for multiprocessing."""
    image_path, output_path, params = args
    status, msg = remove_bg_car_shadow(image_path, output_path, **params)
    return (os.path.basename(image_path), status, msg)


def batch_process_cars(
    input_folder, output_folder,
    feather=3, shadow_gamma=1.5, shadow_darkening=0.75,
    crop=True, padding=20, max_workers=None,
    log_file="processing_log.csv"
):
    os.makedirs(output_folder, exist_ok=True)

    supported_ext = (".jpg", ".jpeg", ".png", ".webp", ".tiff")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]
    if not files:
        print("⚠️ No images found.")
        return

    params = dict(
        feather=feather,
        shadow_gamma=shadow_gamma,
        shadow_darkening=shadow_darkening,
        crop=crop,
        padding=padding
    )

    args_list = [
        (
            os.path.join(input_folder, filename),
            os.path.join(output_folder, os.path.splitext(filename)[0] + "_refined.png"),
            params
        )
        for filename in files
    ]

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(process_file, args_list), total=len(args_list), desc="Processing Cars"):
            results.append(res)

    # Logging CSV
    with open(os.path.join(output_folder, log_file), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Status", "Message"])
        for filename, status, msg in results:
            writer.writerow([filename, "Success" if status else "Failed", msg])

    print(f"\n🎯 Batch processing completed.")
    print(f"📄 Log saved at: {os.path.join(output_folder, log_file)}")


# Example usage
batch_process_cars(
    input_folder="input_cars",
    output_folder="output_cars",
    feather=3, shadow_gamma=1.6, shadow_darkening=0.7,
    crop=True, padding=30,
    max_workers=None,     # None = auto use all CPU cores
    log_file="car_shadow_log.csv"
)
