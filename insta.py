import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import json

# ========= CONFIG =========
background_video_path = "background7.mp4"   # Background looping video
car_image_folder = "Toyota"                  # Folder containing car images
font_path = "LoveDays.ttf"                       # Path to TTF font file
output_video_path = "instagram_reel7.mp4"         # Output file

# Reel constants
reel_width, reel_height = 1080, 1920
fps = 30
slide_duration = 3       # seconds per slide
fade_duration = 0.5      # seconds fade-in/out
# ==========================


# Load car data
with open("car_data.json", "r", encoding="utf-8") as f:
    api_response = json.load(f)

car_data = api_response["data"]

def generate_captions_from_api(api_data):
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
    font = ImageFont.truetype(font_path, font_size)

    words = text.split(" ")
    lines, current_line = [], ""
    max_line_width = width * 0.8

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_line_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    bbox = draw.textbbox((0, 0), "Test", font=font)
    text_height = bbox[3] - bbox[1]
    total_height = len(lines) * text_height + (len(lines) - 1) * 10
    y_offset = 400  # distance from top (you can adjust)

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

def add_outro_text(frame, lines, font_path, start_y=300, font_size=60, line_spacing=20):
    """
    Draws multiple lines (dealer outro) at different vertical positions.
    :param frame: Input frame
    :param lines: List of (text, font_size, color)
    :param font_path: Path to TTF font
    :param start_y: Starting vertical position
    :param line_spacing: Space between lines
    """
    from PIL import ImageFont, ImageDraw, Image

    # Convert frame to PIL for better text rendering
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    width, height = frame_pil.size

    y = start_y
    for text, size, color in lines:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2  # center
        draw.text((x, y), text, font=font, fill=color)
        y += size + line_spacing  # move down

    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


# ---- Load assets ----
background_cap = cv2.VideoCapture(background_video_path)
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (reel_width, reel_height))

car_images = sorted([img for img in os.listdir(car_image_folder) if img.endswith(".png")])
text_data = captions = ['Toyota Aygo 1,0 VVT-i x-cite 5d 2016',
'Pris: 59.900',
'Kan finansieres fra 1.048kr/mdr - uden udbetaling!',
'Ring / SMS for mere info og tid til fremvisning',
'MQM Biler ApS']

# ---- Generate video ----
for i, img_name in enumerate(car_images):
    car_path = os.path.join(car_image_folder, img_name)
    car_image = cv2.imread(car_path, cv2.IMREAD_UNCHANGED)
    if car_image is None:
        print(f"Skipping {car_path}, could not load.")
        continue

    car_resized = resize_and_center(car_image, (reel_width, reel_height))

    # number of frames
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

        # If not the last image, prepare next image for crossfade
        if i < len(car_images) - 1 and frame_idx > total_frames - fade_frames:
            # Load next car
            next_car_path = os.path.join(car_image_folder, car_images[i + 1])
            next_car = cv2.imread(next_car_path, cv2.IMREAD_UNCHANGED)
            next_car_resized = resize_and_center(next_car, (reel_width, reel_height))

            # Generate next frame with same background
            next_frame = overlay_image_alpha(bg_frame.copy(), next_car_resized, 0, 0)
            next_frame = add_multiline_text(next_frame, text_data[min(i+1, len(text_data)-1)], font_path)

            # Alpha between current and next
            alpha = (frame_idx - (total_frames - fade_frames)) / fade_frames
            frame = cv2.addWeighted(frame, 1 - alpha, next_frame, alpha, 0)


        video_writer.write(frame)

# Outro slide (CTA)
# dealer_address = "MQM Biller"
# dealer_mobile = "+45 28495060"
# dealer_website = "www.mqmbiller.dk"

# outro = np.zeros((reel_height, reel_width, 3), dtype=np.uint8)
# # Outro slide
# outro_duration = 3  # seconds
# outro_frames = int(outro_duration * fps)
# outro_lines = [
#     ("Reach us for more detail ...", 80, (255, 215, 0)),   # gold, big
#     (dealer_address, 80, (255, 215, 0)),         # white
#     (f"📞 {dealer_mobile}", 80, (255, 215, 0)),  # white
#     (dealer_website, 80, (255, 215, 0))          # light blue
# ]

# for _ in range(outro_frames):
#     ret, bg_frame = background_cap.read()
#     if not ret:
#         background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         ret, bg_frame = background_cap.read()
#     bg_frame = resize_to_fill(bg_frame, reel_width, reel_height)

#     outro_frame = add_outro_text(bg_frame, outro_lines, font_path)
#     video_writer.write(outro_frame)


background_cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("✅ Instagram Reel generated:", output_video_path)
