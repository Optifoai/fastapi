import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import csv
from ai_video.aws_config import upload_image_bytes_to_s3
import shutil
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

    success, encoded_img = cv2.imencode(".png", result)
    
    if not success:
        return False, None

    return True, encoded_img.tobytes()


def process_file(args):
    """Wrapper for multiprocessing."""
    image_path, output_path, params = args
    status, output= remove_bg_car_shadow(image_path, output_path, **params)
    return (os.path.basename(image_path), status, output)


def batch_process_uploaded_images(
    uploaded_files,
    feather=3, shadow_gamma=1.5, shadow_darkening=0.75,
    crop=True, padding=20, max_workers=None
):
    if not uploaded_files:
        return []

    supported_ext = (".jpg", ".jpeg", ".png", ".webp", ".tiff")

    params = dict(
        feather=feather,
        shadow_gamma=shadow_gamma,
        shadow_darkening=shadow_darkening,
        crop=crop,
        padding=padding
    )

    args_list = []

    temp_folder = "temp_uploads"
    os.makedirs(temp_folder, exist_ok=True)

    # ---- Save uploaded files to temporary local storage ----
    for file in uploaded_files:
        filename = file.filename
        
        if not filename.lower().endswith(supported_ext):
            continue
        
        temp_path = os.path.join(temp_folder, filename)
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        args_list.append((temp_path, None, params))

    results = []
    
    # ---- Process images with multiprocessing ----
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for filename, status, img_data in tqdm(executor.map(process_file, args_list), total=len(args_list)):
            
            if not status:
                results.append({
                    "file": filename,
                    "status": "Failed",
                    "url": None
                })
                continue
            file_bytes = img_data

            # ---- Upload to S3 ----
            object_key = f"processed/{os.path.splitext(filename)[0]}_refined.png"
            url = upload_image_bytes_to_s3(file_bytes, object_key)
            
            results.append({
                "file": filename,
                "status": "Success",
                "url": url
            })

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    return results
