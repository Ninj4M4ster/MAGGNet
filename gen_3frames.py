import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import cv2
import glob

from gen_adn_vae import GenADN_VAE

# --- KONFIGURACJA ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/app/output/maggnet/experiment7_png/checkpoints/maggen_vae_latest.pth"
INPUT_DIR = "/app/data/clean_ct/"
OUTPUT_DIR = "/app/output/3frames_jitter/"
LATENT_DIM = 128

SYNDEEPLEESION_DATASET = True

def apply_window(data, level, width):
    """
    Applies medical windowing to raw image data.
    
    Args:
        data (np.ndarray): Input image data (usually uint16 or int16).
        level (float): Window Center (Level).
        width (float): Window Width.
        
    Returns:
        np.ndarray: Normalized float data ready for uint8 conversion.
    """
    # Handle offset if data is uint16 but represents HU (which are signed)
    # Many 16-bit PNGs store HU with a +32768 offset or simply as unsigned.
    # For generic usage, we operate on the raw values provided.
    
    lower = level - (width / 2)
    upper = level + (width / 2)
    
    # Clip values to the window
    data_clipped = np.clip(data, lower, upper)
    
    # Normalize to 0-255
    if upper == lower:
        return np.zeros_like(data_clipped)
        
    data_normalized = (data_clipped - lower) / (upper - lower) * 255.0
    return data_normalized

def convert_16bit_to_8bit(file_path, window_level=None, window_width=None):
    """
    Reads a 16-bit PNG, applies windowing/normalization, and saves as 8-bit PNG.
    """
    # 1. Load Image
    # We perform standard open to preserve bit-depth initially
    with Image.open(file_path) as img:
        # Convert to NumPy array to handle numerical operations
        # Use int32 to prevent overflow during math operations
        data = np.array(img).astype(np.float32)

    # 2. Process Data
    if window_level is not None and window_width is not None:
        # Apply specific medical windowing
        processed_data = apply_window(data, window_level, window_width)
    else:
        # Auto-scale (Min-Max Normalization)
        # Useful if we just want to see the full dynamic range compressed
        min_val = data.min()
        max_val = data.max()
        
        if max_val == min_val:
            processed_data = np.zeros_like(data)
        else:
            processed_data = (data - min_val) / (max_val - min_val) * 255.0

    # 3. Convert to 8-bit unsigned integer
    image_8bit = processed_data.astype(np.uint8)

    # 4. Create Output Image
    output_img = Image.fromarray(image_8bit)
    
    if output_img.mode != 'L':
        output_img = output_img.convert('L')
    return output_img


def generate_jitter_sequence():
    print("Start")
    for inner_dir in glob.glob(INPUT_DIR + "*"):
        for subdir in glob.glob(inner_dir + "*"):
            subdir_output = OUTPUT_DIR + subdir.split('/')[-1]
            print(f"Przetwarzanie katalogu: {subdir}")
            print(f"Zapisywanie wyników do: {subdir_output}")
            os.makedirs(subdir_output, exist_ok=True)

            model = GenADN_VAE().to(DEVICE)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()

            # frame_names = ["body_img26.png", "body_img27.png", "body_img28.png"]
            frame_names = glob.glob(subdir + "/*.png")
            
            # --- BAZA (STAŁA DLA SEKWENCJI) ---
            z_base = torch.randn(1, LATENT_DIM).to(DEVICE)
            fixed_center = (np.random.randint(200, 312), np.random.randint(220, 340))
            fixed_radius = np.random.randint(6, 12)
            
            num_streaks = np.random.randint(100, 150)
            base_angles = np.random.uniform(0, 360, num_streaks)
            base_intensities = np.random.uniform(0.4, 0.4, num_streaks)

            results = []
            
            with torch.no_grad():
                for i, f_name in enumerate(frame_names):
                    # p_path = os.path.join(INPUT_DIR, f_name)
                    f_last_name = f_name.split("/")[-1]
                    if not os.path.exists(f_name): continue

                    if not SYNDEEPLEESION_DATASET:
                        img_raw = Image.open(f_name).convert('L').resize((512, 512))
                    else:
                        # img_raw = Image.open(f_name)
                        # img_np = np.array(img_raw).astype(np.int32) - 32768
                        # img_raw = Image.fromarray(img_np.astype(np.int32))
                        img_raw = convert_16bit_to_8bit(f_name).resize((512, 512))
                    y = torch.from_numpy(np.array(img_raw)/127.5-1).float().view(1,1,512,512).to(DEVICE)
                    # save_image(y * 0.5 + 0.5, f"{subdir_output}/original_{f_last_name}")
                    
                    # --- JITTER (LEKKIE ZMIANY NA KAŻDEJ KLATCE) ---
                    # 1. Styl: baza + 5% szumu
                    z_jitter = z_base + torch.randn_like(z_base) * 0.05
                    
                    img_np = y.squeeze().cpu().numpy()
                    body_mask = (img_np > -0.5).astype(np.float32)
                    
                    mask_metal = np.zeros((512, 512), dtype=np.uint8)
                    guidance = np.zeros((512, 512), dtype=np.float32)
                    
                    cv2.circle(mask_metal, fixed_center, fixed_radius, 255, -1)
                    
                    # 2. Kąty: każdy kąt przesuwa się o max +/- 0.5 stopnia
                    for angle, intensity in zip(base_angles, base_intensities):
                        jitter_angle = angle + np.random.uniform(-0.5, 0.5)
                        rad = np.deg2rad(jitter_angle)
                        ex = int(fixed_center[0] + 800 * np.cos(rad))
                        ey = int(fixed_center[1] + 800 * np.sin(rad))
                        cv2.line(guidance, fixed_center, (ex, ey), 0.2, 1, cv2.LINE_AA)
                    
                    # guidance = cv2.GaussianBlur(guidance * body_mask, (3, 3), 0)
                    
                    m_t = torch.from_numpy(mask_metal/255.0).view(1,1,512,512).float().to(DEVICE)
                    g_t = torch.from_numpy(guidance).view(1,1,512,512).float().to(DEVICE)
                    y_m = torch.clamp(y + m_t, -1, 1)
                    
                    # Generowanie
                    fake = model.GA(model.EC(y_m, g_t), z_jitter)
                    fake_final = torch.where(torch.from_numpy(body_mask).to(DEVICE) > 0.5, fake, y)
                    
                    results.append(fake_final)
                    save_image(fake_final * 0.5 + 0.5, f"{subdir_output}/jitter_{f_last_name}")

            if results:
                full_seq = torch.cat(results, dim=3)
                save_image(full_seq * 0.5 + 0.5, f"{subdir_output}/sequence_jitter.png")
                print("Sekwencja z lekkim jitterem gotowa.")
            # return

if __name__ == "__main__":
    generate_jitter_sequence()