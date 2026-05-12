import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import cv2
import glob
import re

from gen_adn_vae import GenADN_VAE

# --- KONFIGURACJA ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/app/output/maggnet/experiment_raw/checkpoints/maggen_vae_latest.pth"
INPUT_DIR = "/app/data/Clean_raw/"
OUTPUT_DIR = "/app/output/3frames_jitter_raw_clean_real/"
LATENT_DIM = 128

def read_raw_image(path, target_size=(512, 512), normalize=False):
    """Wczytuje plik .raw i zwraca obraz 2D (H, W)."""
    fname = os.path.basename(path)
    # Ignore files starting with a dot
    if fname.startswith('.'):
        return None
    # Match _H<width>_W<height>
    match = re.search(r'_H(\d+)_W(\d+)', fname)
    if match:
        h_orig, w_orig = int(match.group(1)), int(match.group(2))
    else:
        h_orig, w_orig = 512, 512
        
    try:
        img = np.fromfile(path, dtype=np.float32)
        img = img.reshape((h_orig, w_orig))
    except ValueError:
        # Fallback dla uint16
        img = np.fromfile(path, dtype=np.uint16).astype(np.float32)
        img = img.reshape((h_orig, w_orig))

    if normalize:
        img = np.clip(img, -1024, 3071)
        img = (img + 1024.0) / (3071.0 + 1024.0)  # Shift to [0, 4095]
        img = img * 2.0 - 1.0 # Normalize to [-1, 1]

    if (h_orig, w_orig) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
    return img

def generate_jitter_sequence():
    print("Start")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = GenADN_VAE().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    frame_names = glob.glob(os.path.join(INPUT_DIR, "*.raw"))

    def extract_sort_keys(fname):
        # Example: Clean02_slice0274_H512_W512.raw
        base = os.path.basename(fname)
        m = re.match(r"Clean(\d+)_slice(\d+)_H\d+_W\d+", base)
        if m:
            clean_num = int(m.group(1))
            slice_num = int(m.group(2))
        else:
            clean_num = 0
            slice_num = 0
        return (clean_num, slice_num, base)

    frame_names = sorted(frame_names, key=extract_sort_keys)

    # Initialize random variables
    z_base = torch.randn(1, LATENT_DIM).to(DEVICE)
    fixed_center = (np.random.randint(200, 312), np.random.randint(220, 340))
    fixed_radius = np.random.randint(3, 6)
    num_streaks = np.random.randint(60, 120)
    base_angles = np.random.uniform(0, 360, num_streaks)
    base_intensities = np.random.uniform(1.0, 1.0, num_streaks)

    results = []

    prev_slice_num = None

    with torch.no_grad():
        for i, f_name in enumerate(frame_names):
            # Extract current slice number
            base = os.path.basename(f_name)
            print(base)
            m = re.match(r"Clean(\d+)_slice(\d+)_H\d+_W\d+", base)
            if m:
                curr_slice_num = int(m.group(2))
            else:
                curr_slice_num = None
            # Regenerate random variables if slice number is not consecutive
            if prev_slice_num is not None and curr_slice_num is not None:
                if curr_slice_num != prev_slice_num + 1:
                    z_base = torch.randn(1, LATENT_DIM).to(DEVICE)
                    fixed_center = (np.random.randint(200, 312), np.random.randint(220, 340))
                    fixed_radius = np.random.randint(3, 6)
                    num_streaks = np.random.randint(60, 120)
                    base_angles = np.random.uniform(0, 360, num_streaks)
                    base_intensities = np.random.uniform(1.0, 1.0, num_streaks)
            if curr_slice_num is not None:
                prev_slice_num = curr_slice_num
            f_last_name = os.path.basename(f_name)
            if not os.path.exists(f_name):
                continue
            y_arr = read_raw_image(f_name, normalize=True)
            if y_arr is None:
                continue
            y = torch.from_numpy(y_arr).float().unsqueeze(0).to(DEVICE)
            # save_image(y * 0.5 + 0.5, f"{OUTPUT_DIR}/original_{f_last_name}")

            # --- JITTER (LEKKIE ZMIANY NA KAŻDEJ KLATCE) ---
            # 1. Styl: baza + 5% szumu
            z_jitter = z_base + torch.randn_like(z_base) * 0.05

            img_np = y.squeeze().cpu().numpy()
            body_mask = (img_np > -0.98).astype(np.float32)

            mask_metal = np.zeros((512, 512), dtype=np.uint8)
            guidance = np.zeros((512, 512), dtype=np.float32)

            cv2.circle(mask_metal, fixed_center, fixed_radius, 255, -1)

            # 2. Kąty: każdy kąt przesuwa się o max +/- 0.5 stopnia
            for angle, intensity in zip(base_angles, base_intensities):
                jitter_angle = angle + np.random.uniform(-0.5, 0.5)
                rad = np.deg2rad(jitter_angle)
                ex = int(fixed_center[0] + 800 * np.cos(rad))
                ey = int(fixed_center[1] + 800 * np.sin(rad))
                cv2.line(guidance, fixed_center, (ex, ey), 0.6, 1, cv2.LINE_AA)

            # guidance = cv2.GaussianBlur(guidance * body_mask, (3, 3), 0)

            m_t = torch.from_numpy(mask_metal/255.0).view(1,1,512,512).float().to(DEVICE)
            g_t = torch.from_numpy(guidance).view(1,1,512,512).float().to(DEVICE)
            y_m = torch.clamp(y + m_t, -1, 1)

            # Generowanie
            fake = model.GA(model.EC(y_m, g_t), z_jitter)
            fake_final = torch.where(torch.from_numpy(body_mask).to(DEVICE) > 0.5, fake, y)

            results.append(fake_final)
            # Save as raw with reversed scaling
            fake_np = fake_final.squeeze().cpu().numpy()
            # Reverse normalization: [-1,1] -> [0,1] -> [HU]
            fake_np_rescaled = (fake_np + 1.0) / 2.0  # [0,1]
            fake_np_rescaled = fake_np_rescaled * (3071.0 + 1024.0) - 1024.0  # [-1024,3071]
            # Save as float32 raw
            raw_out_path = os.path.join(OUTPUT_DIR, f"jitter_{os.path.splitext(f_last_name)[0]}.raw")
            fake_np_rescaled.astype(np.float32).tofile(raw_out_path)

    if results:
        full_seq = torch.cat(results, dim=3)
        # Save only the first two images from the sequence
        save_image(results[0] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part.png")
        save_image(results[1] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part1.png")
        save_image(results[2] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part2.png")
        save_image(results[3] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part3.png")
        save_image(results[4] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part4.png")
        save_image(results[5] * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter_part5.png")
        save_image(full_seq * 0.5 + 0.5, f"{OUTPUT_DIR}/sequence_jitter.png")
        print("Sekwencja z lekkim jitterem gotowa.")

if __name__ == "__main__":
    generate_jitter_sequence()