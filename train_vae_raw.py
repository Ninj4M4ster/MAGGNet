import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import os, cv2, numpy as np, itertools, glob, re
from tqdm import tqdm
from PIL import Image

from gen_adn_vae import GenADN_VAE

DATA_DIR = "/app/data/RPI/body"

OUTPUT_PATH = "/app/output/maggnet/experiment_raw"
os.makedirs(OUTPUT_PATH, exist_ok=True)

SAMPLES_DIR = os.path.join(OUTPUT_PATH, "samples_vae")
CHECKPOINTS_DIR = os.path.join(OUTPUT_PATH, "checkpoints")

# Foldery wykluczone z treningu (Test Set)
TEST_FOLDERS = ['body2', 'body3', 'body4', 'body7', 'body8', 'body9', 'body10', 'body11', 'body12', 'body13', 'head1', 'head2']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HIPERPARAMETRY ---
LR = 1e-4
LAMBDA_CONTENT = 30.0
LAMBDA_EDGE = 1.5
LAMBDA_BG = 100.0
LAMBDA_METAL = 1000.0
LAMBDA_KLD = 0.005
LAMBDA_GUIDANCE = 10.0
EPOCHS = 100

def read_raw_image(path, target_size=(512, 512), normalize=False):
    """Wczytuje plik .raw i zwraca obraz 2D (H, W)."""
    fname = os.path.basename(path)
    match = re.search(r'[x_](\d+)x(\d+)', fname)
    
    if match:
        w_orig, h_orig = int(match.group(1)), int(match.group(2))
    else:
        w_orig, h_orig = 512, 512
        
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

def save_raw_image(tensor, path):
    """Zapisuje tensor PyTorch jako plik binarny .raw (float32)."""
    img = tensor.squeeze().detach().cpu()
    img_np = img.numpy().astype(np.float32)

    # Ensure output is 2D
    if img_np.ndim == 3:
        img_np = img_np[0]

    target_size = (512, 512)
    if img_np.shape != target_size:
        # Upsample to target size using cv2
        img_np = cv2.resize(img_np, target_size, interpolation=cv2.INTER_CUBIC)

    # Save denormalized version in .raw format (-1024 to 3071)
    img_denorm = (img_np * 0.5 + 0.5) * (3071.0 + 1024.0) - 1024.0
    img_denorm = np.clip(img_denorm, -1024, 3071).astype(np.float32)
    img_denorm.tofile(path.replace('.png', '.raw'))

    # Save normalized version in .png format
    img_png = (img_np * 0.5 + 0.5) * 255
    img_png = np.clip(img_png, 0, 255).astype(np.uint8)
    png_path = path.replace('.raw', '.png')
    cv2.imwrite(png_path, img_png)
    img_np.tofile(path)

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
        body_limit = (img > -0.95).astype(np.float32)

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
                cv2.fillPoly(mask_metal, [pts], 200)
            else:
                radius = 2
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
                    intensity = 0.6

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

                        intensity = 0.2
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
        img_with_metal = img * (1 - mask_float) + mask_float * 1.0
        final_g = np.clip(guidance + mask_float, 0.0, 1.0)

        imgs_out.append(torch.from_numpy(img_with_metal).unsqueeze(0))
        guides_out.append(torch.from_numpy(final_g).unsqueeze(0))
        masks_out.append(torch.from_numpy(mask_float).unsqueeze(0))

    img_with_metal = torch.stack(imgs_out, dim=0).to(device).float()
    final_guidance = torch.stack(guides_out, dim=0).to(device).float()
    mask_metal = torch.stack(masks_out, dim=0).to(device).float()

    return img_with_metal, final_guidance, mask_metal

def collect_training_files(base_dir, test_folders_list):
    """
    Przeszukuje podfoldery Body/Head wewnątrz kontenera.
    """
    all_targets = []
    all_arts = []
    
    print(f"Szukanie danych w kontenerze: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"BŁĄD: Ścieżka {base_dir} nie istnieje w kontenerze! Sprawdź bindy w .sh")
        return [], [], []

    subfolders = sorted(os.listdir(base_dir))
    for folder_name in subfolders:
        folder_full_path = os.path.join(base_dir, folder_name)
        print(folder_name)
        
        if not os.path.isdir(folder_full_path):
            continue
            
        if folder_name in test_folders_list:
            print(f" -> Pomijanie folderu testowego: {folder_name}")
            continue

        dir_free = os.path.join(folder_full_path, "Target")
        dir_art = os.path.join(folder_full_path, "Baseline")
        
        if not (os.path.exists(dir_free) and os.path.exists(dir_art)):
            continue

        f_free = sorted(glob.glob(os.path.join(dir_free, "*img*.raw")))
        f_art = sorted(glob.glob(os.path.join(dir_art, "*img*.raw")))
        
        min_len = min(len(f_free), len(f_art))
        
        if min_len > 0:
            all_targets.extend(f_free[:min_len])
            all_arts.extend(f_art[:min_len])

    return all_targets, all_arts

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, target_files, art_files):
        self.target_files = target_files
        self.art_files = art_files

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, idx):
        y = torch.from_numpy(read_raw_image(self.target_files[idx], normalize=True)).float().unsqueeze(0)
        x_a = torch.from_numpy(read_raw_image(self.art_files[idx], normalize=True)).float().unsqueeze(0)

        return y, x_a

def total_variation_loss(img, mask=None):
    # img: (B, 1, H, W)
    # mask: (B, 1, H, W) or None
    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    if mask is not None:
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        tv = (diff_x**2 * mask_x).mean() + (diff_y**2 * mask_y).mean()
    else:
        tv = (diff_x**2).mean() + (diff_y**2).mean()
    return tv

    return loss.mean()

def train():
    # Upewnij się, że foldery wyjściowe istnieją (wewnątrz podmontowanego wolumenu)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    print(f"Start treningu na urządzeniu: {DEVICE}")
    # print(f"Wyniki będą zapisywane w: {OUTPUT_DIR}")

    model = GenADN_VAE().to(DEVICE)
    opt_G = optim.Adam(itertools.chain(model.EC.parameters(), model.EA.parameters(), model.GA.parameters()), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(model.DA.parameters(), lr=LR, betas=(0.5, 0.999))
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=DEVICE).float().view(1,1,3,3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=DEVICE).float().view(1,1,3,3)

    # --- ZBIERANIE DANYCH ---
    files_target, files_art = collect_training_files(DATA_DIR, TEST_FOLDERS)
    dataset = CustomDataset(files_target, files_art)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=6)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    if len(files_target) == 0:
        print("BŁĄD KRYTYCZNY: Nie znaleziono żadnych plików treningowych! Sprawdź ścieżki.")
        # Warto rzucić wyjątek, żeby job w SLURM zakończył się błędem, a nie sukcesem
        raise RuntimeError("Brak danych treningowych.")

    print(f"Łącznie zebrano {len(files_target)} trójek treningowych.")

    for epoch in range(EPOCHS):
        i = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}", mininterval=10.0)
        for y, x_a in loop:
            y = y.to(DEVICE)
            x_a = x_a.to(DEVICE)

            y_m, g, m = get_advanced_guidance(y, max_metals=2, irregular_shapes=False)
            # --- G STEP ---
            opt_G.zero_grad()
            z, mu, logvar = model.EA(x_a)
            fake = model.GA(model.EC(y_m, g), z)

            body_mask = (y > -0.95).float()
            # Wymuszanie czystego tła y
            fake_final = torch.where(body_mask > 0.5, fake, y)

            loss_GAN = F.mse_loss(model.DA(fake_final), torch.ones_like(model.DA(fake_final)))
            content_mask = (body_mask * (g < 0.2)).float()
            loss_content = F.l1_loss(fake_final * content_mask, y * content_mask) * LAMBDA_CONTENT
            loss_bg = F.mse_loss(fake * (1 - body_mask), y * (1 - body_mask)) * LAMBDA_BG
            
            # Edge Loss
            # gx = F.conv2d(fake_final, sobel_x, padding=1)
            # gy = F.conv2d(fake_final, sobel_y, padding=1)
            # loss_edge = -torch.mean(torch.sqrt(gx**2 + gy**2 + 1e-8) * (g > 0.1).float()) * LAMBDA_EDGE
            
            gx = F.conv2d(fake_final, sobel_x, padding=1)
            gy = F.conv2d(fake_final, sobel_y, padding=1)
            gx_g = F.conv2d(g, sobel_x, padding=1)
            gy_g = F.conv2d(g, sobel_y, padding=1)
            loss_edge = F.l1_loss(torch.sqrt(gx**2 + gy**2 + 1e-8), torch.sqrt(gx_g**2 + gy_g**2 + 1e-8)) * LAMBDA_EDGE
            
            # VAE KLD Loss
            loss_KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * LAMBDA_KLD

            loss_metal_shape = F.l1_loss(
                (fake > 0.85).float(),
                m
            ) * LAMBDA_METAL

            loss_guidance = -torch.mean(fake_final * g) * LAMBDA_GUIDANCE
            
            total_G = loss_GAN + loss_content + loss_edge + loss_bg + loss_metal_shape + loss_KLD + loss_guidance
            total_G.backward()
            opt_G.step()

            # --- D STEP ---
            opt_D.zero_grad()
            loss_D = (F.mse_loss(model.DA(x_a), torch.ones_like(model.DA(x_a))) + 
                      F.mse_loss(model.DA(fake_final.detach()), torch.zeros_like(model.DA(fake_final)))) * 0.5
            loss_D.backward()
            opt_D.step()

            if i % 100 == 0:
                preview = torch.cat([y, g*2-1, x_a, fake_final, y_m, fake > 0.85, m], dim=3)
                save_image(preview * 0.5 + 0.5, f"{OUTPUT_PATH}/samples_vae/e{epoch}_{i}.png")
                print(f"loss_GAN: {loss_GAN.item():.6f}, loss_content: {loss_content.item():.6f}, loss_edge: {loss_edge.item():.6f}, loss_bg: {loss_bg.item():.6f}, loss_METAL: {loss_metal_shape.item():.6f}, loss_KLD: {loss_KLD.item():.6f}, loss_guidance: {loss_guidance.item():.6f}, loss_D: {loss_D.item():.6f}")
                # import matplotlib.pyplot as plt
                # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # axes[0].hist(x_a.cpu().detach().numpy().flatten(), bins=100, alpha=0.7)
                # axes[0].set_title('x_a pixel values')
                # axes[0].set_xlabel('Pixel value')
                # axes[0].set_ylabel('Frequency')
                # axes[1].hist(y.cpu().detach().numpy().flatten(), bins=100, alpha=0.7)
                # axes[1].set_title('y pixel values')
                # axes[1].set_xlabel('Pixel value')
                # axes[1].set_ylabel('Frequency')
                # plt.tight_layout()
                # plt.savefig(f"{OUTPUT_PATH}/samples_vae/hist_e{epoch}_{i}.png")
                # plt.close()
                # loop.set_postfix(G=total_G.item(), KLD=loss_KLD.item())
            i += 1

        torch.save(model.state_dict(), f"{OUTPUT_PATH}/checkpoints/maggen_vae_latest.pth")

if __name__ == "__main__":
    train()