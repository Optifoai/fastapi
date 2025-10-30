import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import urllib.request
import shutil
import subprocess
import datetime
from ai_video.aws_config import *  # your S3 upload config

# ========= CONFIG =========
background_video_path = "ai_video/background.mp4"
font_path = "ai_video/LoveDays.ttf"
audio_file_path = "ai_video/music.mp3"

# Reel constants
reel_width, reel_height = 1080, 1920
fps = 30
slide_duration = 3
fade_duration = 0.5
# ==========================

# Ensure public folder exists
os.makedirs("public/videos", exist_ok=True)


# ========== HELPERS ==========

def download_image(url, save_path):
    """Download image from URL."""
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def overlay_image_alpha(background, overlay, x, y):
    """Overlay RGBA image onto BGR background."""
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
    """Resize car image and center on canvas."""
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
    """Add centered multiline text."""
    width, height = frame.shape[1], frame.shape[0]
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay_pil = Image.new("RGBA", frame_pil.size, (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay_pil)
    font = ImageFont.truetype(font_path, font_size)

    words = text.split(" ")
    lines, current_line = [], []
    max_line_width = width * 0.8

    for word in words:
        test_line = current_line + [word]
        test_string = " ".join(test_line)
        bbox = draw.textbbox((0, 0), test_string, font=font)
        if bbox[2] - bbox[0] > max_line_width:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))

    bbox = draw.textbbox((0, 0), "Test", font=font)
    text_height = bbox[3] - bbox[1]
    total_height = len(lines) * text_height + (len(lines) - 1) * 10
    y_offset = 400

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


# ========== MAIN VIDEO FUNCTION ==========

def generate_video_with_audio(car_images, text_data, temp_folder):
    """Generate video and merge with background audio using FFmpeg."""
    background_cap = cv2.VideoCapture(background_video_path)
    if not background_cap.isOpened():
        raise Exception("Could not open background video")

    temp_video_path = "temp_video.mp4"
    video_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (reel_width, reel_height))

    if not video_writer.isOpened():
        raise Exception("Could not create video writer")

    # Generate video frames
    for i, img_name in enumerate(car_images):
        car_path = os.path.join(temp_folder, img_name)
        car_image = cv2.imread(car_path, cv2.IMREAD_UNCHANGED)
        if car_image is None:
            print(f"⚠️ Could not load image: {car_path}")
            continue

        car_resized = resize_and_center(car_image, (reel_width, reel_height))
        total_frames = int(slide_duration * fps)
        fade_frames = int(fade_duration * fps)

        for frame_idx in range(total_frames):
            ret, bg_frame = background_cap.read()
            if not ret:
                background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bg_frame = background_cap.read()
            bg_frame = resize_to_fill(bg_frame, reel_width, reel_height)

            frame = overlay_image_alpha(bg_frame.copy(), car_resized, 0, 0)
            frame = add_multiline_text(frame, text_data[min(i, len(text_data) - 1)], font_path)

            # Crossfade transition
            if i < len(car_images) - 1 and frame_idx > total_frames - fade_frames:
                next_car_path = os.path.join(temp_folder, car_images[i + 1])
                next_car = cv2.imread(next_car_path, cv2.IMREAD_UNCHANGED)
                if next_car is not None:
                    next_car_resized = resize_and_center(next_car, (reel_width, reel_height))
                    next_frame = overlay_image_alpha(bg_frame.copy(), next_car_resized, 0, 0)
                    next_frame = add_multiline_text(next_frame, text_data[min(i + 1, len(text_data) - 1)], font_path)
                    alpha = (frame_idx - (total_frames - fade_frames)) / fade_frames
                    frame = cv2.addWeighted(frame, 1 - alpha, next_frame, alpha, 0)

            video_writer.write(frame)

    background_cap.release()
    video_writer.release()

    # --- Merge Audio with FFmpeg ---
    if not os.path.exists(audio_file_path):
        raise Exception(f"Audio file not found: {audio_file_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = f"car_reel_{timestamp}.mp4"

    print("🎵 Adding background audio with FFmpeg...")

    # Check FFmpeg availability
    if shutil.which("ffmpeg") is None:
        raise Exception("FFmpeg is not installed or not in PATH. Please install it to merge audio.")

    # Command: loop audio, set volume 70%, and cut to video length
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-stream_loop", "-1", "-i", audio_file_path,
        "-shortest",
        "-filter:a", "volume=0.7",
        "-c:v", "copy",
        "-c:a", "aac",
        final_video_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Audio added successfully with FFmpeg!")
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg failed to combine video and audio:\n{e.stderr.decode()}")

    os.remove(temp_video_path)
    return final_video_path


# ========== MAIN REEL PROCESSOR ==========

def process_car_reel(image_urls, text_data):
    """Main function to process car reel with images and text."""
    if not image_urls:
        raise Exception("No image URLs provided")

    if not text_data:
        raise Exception("No text data provided")

    temp_folder = "temp_images"
    os.makedirs(temp_folder, exist_ok=True)

    try:
        # Download or reuse cached images
        car_images = []
        for i, url in enumerate(image_urls):
            img_path = os.path.join(temp_folder, f"car_{i}.png")
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                print(f"🟢 Using cached image: {img_path}")
                car_images.append(f"car_{i}.png")
                continue

            if download_image(url, img_path):
                print(f"✅ Downloaded image: {url}")
                car_images.append(f"car_{i}.png")
            else:
                print(f"❌ Failed to download image: {url}")

        if not car_images:
            raise Exception("No images downloaded successfully or found in cache")

        print(f"🎬 Generating video with {len(car_images)} images...")
        final_video_path = generate_video_with_audio(car_images, text_data, temp_folder)

        print("📤 Uploading video to S3...")
        s3_url = upload_video_to_s3(final_video_path, "ai_videos")

        if os.path.exists(final_video_path):
            os.remove(final_video_path)

        return {
            "status": "success",
            "message": "Video generated and uploaded to S3 successfully",
            "s3_url": s3_url,
            "images_processed": len(car_images),
            "text_used": text_data
        }

    finally:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
