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
INPUT_DIR = "/app/data/clean_ct/images_png_56/"
OUTPUT_DIR = "/app/output/random_jitter/"
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

def get_advanced_guidance(img_tensor, max_metals=2, irregular_shapes=False):
    """
    Batch-aware version.
    
    Args:
        img_tensor: torch.Tensor of shape (B, 1, H, W) or (B, H, W)
    Returns:
        img_with_metal: (B, 1, H, W)
        final_guidance: (B, 1, H, W)
        mask_metal:     (B, 1, H, W)
    """
    if img_tensor.dim() == 3:
        # (B, H, W) -> (B, 1, H, W)
        img_tensor = img_tensor.unsqueeze(1)

    assert img_tensor.dim() == 4, "Expected input shape (B, 1, H, W)"

    B, _, H, W = img_tensor.shape
    device = img_tensor.device

    imgs_out = []
    guides_out = []
    masks_out = []

    for b in range(B):
        img = img_tensor[b, 0].cpu().numpy()

        # Detekcja sylwetki pacjenta
        body_limit = (img > -0.9).astype(np.float32)

        guidance = np.zeros((H, W), dtype=np.float32)
        mask_metal = np.zeros((H, W), dtype=np.uint8)

        num_metals = np.random.randint(1, max_metals + 1)
        metal_centers = []

        for _ in range(num_metals):
            center = (
                np.random.randint(180, min(332, W)),
                np.random.randint(180, min(332, H))
            )
            metal_centers.append(center)

            if irregular_shapes:
                num_pts = np.random.randint(4, 9)
                pts = []
                for i in range(num_pts):
                    dist = np.random.randint(5, 15)
                    angle = (i / num_pts) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
                    px = int(center[0] + dist * np.cos(angle))
                    py = int(center[1] + dist * np.sin(angle))
                    pts.append([px, py])

                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask_metal, [pts], 255)
            else:
                radius = np.random.randint(3, 9)
                cv2.circle(mask_metal, center, radius, 255, -1)
            

            num_streaks = np.random.randint(30, 120)

            for _ in range(num_streaks):
                angle = np.random.uniform(0, 2 * np.pi)
                length = 800
                thickness = 1  # force ultra-thin streaks

                # number of segments controls smoothness of decay
                n_segments = 100

                for i in range(n_segments):
                    t0 = i / n_segments
                    t1 = (i + 1) / n_segments

                    # linear decay (can be swapped for exponential)
                    intensity = 0.2

                    if intensity <= 0:
                        break

                    x0 = int(center[0] + t0 * length * np.cos(angle))
                    y0 = int(center[1] + t0 * length * np.sin(angle))
                    x1 = int(center[0] + t1 * length * np.cos(angle))
                    y1 = int(center[1] + t1 * length * np.sin(angle))

                    cv2.line(
                        guidance,
                        (x0, y0),
                        (x1, y1),
                        intensity,
                        thickness,
                        cv2.LINE_AA
                    )
        for i in range(len(metal_centers)):
            for j in range(i + 1, len(metal_centers)):
                x1, y1 = metal_centers[i]
                x2, y2 = metal_centers[j]

                angle = np.arctan2(y2 - y1, x2 - x1)

                # draw both directions
                for sign in (-1, 1):
                    ang = angle + sign * 0.0  # no jitter yet

                    length = 900
                    n_segments = 70

                    for k in range(n_segments):
                        t0 = k / n_segments
                        t1 = (k + 1) / n_segments

                        intensity = np.exp(-2.2 * t0)
                        if intensity < 0.05:
                            break

                        cx = int(x1 + sign * t0 * length * np.cos(ang))
                        cy = int(y1 + sign * t0 * length * np.sin(ang))
                        nx = int(x1 + sign * t1 * length * np.cos(ang))
                        ny = int(y1 + sign * t1 * length * np.sin(ang))

                        if 0 <= cx < W and 0 <= cy < H:
                            if body_limit[cy, cx] > 0:
                                cv2.line(
                                    guidance,
                                    (cx, cy),
                                    (nx, ny),
                                    intensity,
                                    thickness=3,
                                    lineType=cv2.LINE_AA
                                )

        guidance = guidance * body_limit
        # guidance = cv2.GaussianBlur(guidance, (3, 3), sigmaX=0.35)

        mask_float = mask_metal.astype(np.float32) / 255.0
        img_with_metal = np.clip(img + mask_float, -1.0, 1.0)
        final_g = np.clip(guidance + mask_float, 0.0, 1.0)

        imgs_out.append(torch.from_numpy(img_with_metal).unsqueeze(0))
        guides_out.append(torch.from_numpy(final_g).unsqueeze(0))
        masks_out.append(torch.from_numpy(mask_float).unsqueeze(0))

    img_with_metal = torch.stack(imgs_out, dim=0).to(device).float()
    final_guidance = torch.stack(guides_out, dim=0).to(device).float()
    mask_metal = torch.stack(masks_out, dim=0).to(device).float()

    return img_with_metal, final_guidance, mask_metal


def generate_jitter_sequence():
    print("Start")
    for subdir in glob.glob(INPUT_DIR + "*"):
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
                
                # --- JITTER (CAŁKOWICIE LOSOWY DLA KAŻDEJ KLATKI) ---
                # 1. Styl: baza + 5% szumu (pozostaje, ale można też zrobić całkowicie losowy z_jitter)
                z_jitter = torch.randn_like(z_base)  # całkowicie losowy styl

                img_np = y.squeeze().cpu().numpy()
                body_mask = (img_np > -0.5).astype(np.float32)

                # Losowe centrum i promień metalu dla każdej klatki
                random_center = (np.random.randint(200, 312), np.random.randint(220, 340))
                random_radius = np.random.randint(6, 12)

                mask_metal = np.zeros((512, 512), dtype=np.uint8)
                guidance = np.zeros((512, 512), dtype=np.float32)

                cv2.circle(mask_metal, random_center, random_radius, 255, -1)

                # Losowa liczba streaków, kąty i intensywności dla każdej klatki
                num_streaks = np.random.randint(100, 150)
                angles = np.random.uniform(0, 360, num_streaks)
                intensities = np.random.uniform(0.2, 0.4, num_streaks)

                for angle, intensity in zip(angles, intensities):
                    jitter_angle = angle + np.random.uniform(-5, 5)  # większy jitter
                    rad = np.deg2rad(jitter_angle)
                    ex = int(random_center[0] + 800 * np.cos(rad))
                    ey = int(random_center[1] + 800 * np.sin(rad))
                    cv2.line(guidance, random_center, (ex, ey), float(intensity), 1, cv2.LINE_AA)
                
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