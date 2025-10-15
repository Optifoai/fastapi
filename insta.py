import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import json
import shutil
import uuid

# ========= CONFIG =========
def create_dynamic_output_path(base_folder="generated_videos", base_name="insta_reel", extension=".mp4"):
    os.makedirs(base_folder, exist_ok=True)
    i = 1
    while True:
        file_name = f"{base_name}_{i}{extension}"
        file_path = os.path.join(base_folder, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

# Reel constants
reel_width, reel_height = 1080, 1920
fps = 30
slide_duration = 3       # seconds per slide
fade_duration = 0.5      # seconds fade-in/out
# ==========================

def save_images_to_folder(image_paths, folder_name="car_images"):
    """Save images to temporary folder - FIXED for AWS compatibility"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Copy all images into the folder
    for img_path in image_paths:
        if os.path.isfile(img_path):
            shutil.copy(img_path, folder_name)

    return folder_name  # Return relative path for AWS compatibility

def generate_captions_from_api(api_data):
    """Generate captions from API data"""
    if not api_data:
        return ["Car Details", "No information available"]
    
    captions = []
    brand = api_data.get("brand", "").title()
    model = api_data.get("model", "").title()
    year = api_data.get("model_year", "")
    version = api_data.get("version", "")

    captions.append(f"{year} {brand} {model} {version}")

    fuel = api_data.get("fuel_type", "")
    power = api_data.get("engine_power", "")
    efficiency = api_data.get("fuel_efficiency", "")
    if fuel or power or efficiency:
        captions.append(f"Fuel: {fuel} | {power} HP | {efficiency} km/l")

    body = api_data.get("body_type", {}).get("name", "")
    color = api_data.get("color", {}).get("name", "")
    seats = api_data.get("minimum_seats", "")
    captions.append(f"{body} | {color} | {seats} seats")

    speed = api_data.get("top_speed", None)
    if speed:
        captions.append(f"Top Speed: {speed} km/h")

    # Select top 3 equipment items
    if api_data.get("equipment"):
        eq_list = [eq["name"] for eq in api_data["equipment"] if eq["quantity"] > 0]
        if eq_list:
            captions.append(" • ".join(eq_list[:3]))

    captions.append("Available now 🚗✨")
    return captions

def overlay_image_alpha(background, overlay, x, y):
    """ Overlay RGBA image onto BGR background. """
    overlay_h, overlay_w, _ = overlay.shape
    bg_h, bg_w, _ = background.shape

    if x >= bg_w or y >= bg_h:
        return background
    if x + overlay_w > bg_w:
        overlay_w = bg_w - x
    if y + overlay_h > bg_h:
        overlay_h = bg_h - y

    overlay_region = overlay[:overlay_h, :overlay_w]
    alpha_channel = overlay_region[:, :, 3] / 255.0
    alpha_channel = cv2.merge((alpha_channel, alpha_channel, alpha_channel))

    background_region = background[y:y + overlay_h, x:x + overlay_w]
    blended_region = alpha_channel * overlay_region[:, :, :3] + (1 - alpha_channel) * background_region
    background[y:y + overlay_h, x:x + overlay_w] = blended_region.astype(np.uint8)

    return background

def resize_and_center(image, target_size):
    """Resize car image with transparency and center inside canvas."""
    target_width, target_height = target_size
    h, w = image.shape[:2]

    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if resized_image.shape[2] == 3:
        alpha_channel = np.ones((new_h, new_w), dtype=np.uint8) * 255
        resized_image = np.dstack((resized_image, alpha_channel))

    new_image = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    new_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return new_image

def add_multiline_text(frame, text, font_path, font_size=60, color=(0, 0, 0)):
    """Add centered multiline text with TTF font."""
    width, height = frame.shape[1], frame.shape[0]
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay_pil = Image.new("RGBA", frame_pil.size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay_pil)
    
    # Handle font loading error
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()

    words = text.split(" ")
    lines, current_line = [], []
    max_line_width = width * 0.8

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_line_width and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))

    bbox = draw.textbbox((0, 0), "Test", font=font)
    text_height = bbox[3] - bbox[1]
    total_height = len(lines) * text_height + (len(lines) - 1) * 10
    y_offset = 400  # distance from top

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y_offset), line, font=font, fill=color + (255,))
        y_offset += text_height + 10

    blended = Image.alpha_composite(frame_pil, overlay_pil)
    return cv2.cvtColor(np.array(blended.convert("RGB")), cv2.COLOR_RGB2BGR)

def resize_to_fill(image, target_width, target_height):
    """Resize background to fill screen (crop if needed)."""
    h, w, _ = image.shape
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_x = (new_w - target_width) // 2
    crop_y = (new_h - target_height) // 2
    return resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

def delete_all_files(folder_path: str):
    """Deletes all files inside the given folder."""
    if not os.path.exists(folder_path):
        return

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def main(car_image_list, background_video_path="background.mp4", font_path="LoveDays.ttf"):
    """Main video generation function - FIXED for API integration"""
    
    # Create unique output path
    output_video_path = create_dynamic_output_path()
    
    # Load car data from JSON (you might want to pass this as parameter instead)
    try:
        with open("car_data.json", "r", encoding="utf-8") as f:
            api_response = json.load(f)
        car_data = api_response.get("data", {})
        # Generate captions from API data
        text_data = generate_captions_from_api(car_data)
    except Exception as e:
        print(f"Warning: Could not load car data: {e}")
        # Fallback captions
        text_data = ["Car Details", "Information not available", "Contact for more info"]
    
    print(f"Processing {len(car_image_list)} car images")
    
    # Save images to temporary folder
    car_image_folder = save_images_to_folder(car_image_list, folder_name="car_images")
    
    # Load background video
    background_cap = cv2.VideoCapture(background_video_path)
    if not background_cap.isOpened():
        raise Exception(f"Could not open background video: {background_video_path}")
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (reel_width, reel_height))

    # Get car images
    car_images = sorted([img for img in os.listdir(car_image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Generate video slides
    for i, img_name in enumerate(car_images):
        car_path = os.path.join(car_image_folder, img_name)
        car_image = cv2.imread(car_path, cv2.IMREAD_UNCHANGED)
        if car_image is None:
            print(f"Skipping {car_path}, could not load.")
            continue

        car_resized = resize_and_center(car_image, (reel_width, reel_height))

        # Number of frames per slide
        total_frames = int(slide_duration * fps)
        fade_frames = int(fade_duration * fps)

        for frame_idx in range(total_frames):
            ret, bg_frame = background_cap.read()
            if not ret:  # loop background
                background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bg_frame = background_cap.read()
            bg_frame = resize_to_fill(bg_frame, reel_width, reel_height)

            frame = overlay_image_alpha(bg_frame.copy(), car_resized, 0, 0)
            frame = add_multiline_text(frame, text_data[min(i, len(text_data)-1)], font_path)

            # Crossfade to next image
            if i < len(car_images) - 1 and frame_idx > total_frames - fade_frames:
                next_car_path = os.path.join(car_image_folder, car_images[i + 1])
                next_car = cv2.imread(next_car_path, cv2.IMREAD_UNCHANGED)
                if next_car is not None:
                    next_car_resized = resize_and_center(next_car, (reel_width, reel_height))
                    next_frame = overlay_image_alpha(bg_frame.copy(), next_car_resized, 0, 0)
                    next_frame = add_multiline_text(next_frame, text_data[min(i+1, len(text_data)-1)], font_path)
                    alpha = (frame_idx - (total_frames - fade_frames)) / fade_frames
                    frame = cv2.addWeighted(frame, 1 - alpha, next_frame, alpha, 0)

            video_writer.write(frame)

    # Cleanup
    background_cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Delete temporary car images
    delete_all_files(car_image_folder)
    if os.path.exists(car_image_folder):
        os.rmdir(car_image_folder)
    
    print("✅ Instagram Reel generated:", output_video_path)
    return output_video_path