import torch
import torch.nn as nn
import numpy as np
import os
import glob
import cv2
import pandas as pd
import re
from tqdm import tqdm
from piq import fsim, psnr, ssim, multi_scale_ssim
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image

try:
    from gen_adn_advanced import GenADN_Final
    from gen_adn_vae import GenADN_VAE
except ImportError:
    print("BŁĄD: Nie znaleziono plików gen_adn_advanced.py lub gen_adn_vae.py!")
    exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = "/app/output/RPI/body5"
INFERENCE_ONLY = True
TEST_DIR_PARAMETRIZED = lambda i: f"/app/output/RPI/body{i}"
OUTPUT_CSV = "/app/output/experiments_output/eval_2_metals_20-100_streaks/evaluation_results.csv"
OUTPUT_DIR = "/app/output/experiments_output/body_generation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# standard or vae
MODEL_TYPE = "vae" 
# raw or png
CHECKPOINT_PATH = "/app/output/maggnet/experiment2/checkpoints/maggen_vae_latest.pth"


INPUT_FORMAT = "png" 

"""
Żeby użyć StyleLoss trzeba wagi VGG mieć w Cache
Komenda: cp $PLG_GROUPS_STORAGE/plggsmartudlteam/maggnet/vgg19-dcbb9e9d.pth ~/.cache/torch/hub/checkpoints/
"""
class StyleLossMetric(nn.Module):
    def __init__(self, device):
        super().__init__()
        print("Ładowanie VGG19 do metryki stylu...")
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się pobrać wag VGG (brak internetu?). Błąd: {e}")
            self.model = None
            return

        self.style_layers = [0, 5, 10, 19, 28] 
        self.model = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def gram_matrix(self, input):
        a, b, c, d = input.size() 
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t()) 
        return G.div(a * b * c * d)

    def forward(self, gen, real):
        if self.model is None: return float('nan')

        if gen.shape[1] == 1:
            gen = gen.repeat(1, 3, 1, 1)
            real = real.repeat(1, 3, 1, 1)
            
        gen = (gen - self.mean) / self.std
        real = (real - self.mean) / self.std

        style_loss = 0
        x_gen, x_real = gen, real
        
        for i, layer in enumerate(self.model):
            x_gen = layer(x_gen)
            x_real = layer(x_real)
            
            if i in self.style_layers:
                gm_gen = self.gram_matrix(x_gen)
                gm_real = self.gram_matrix(x_real)
                style_loss += torch.nn.functional.mse_loss(gm_gen, gm_real)
                
        return style_loss.item()

# --- FUNKCJE POMOCNICZE ---

def read_raw_file(path, target_size):
    """Logika wczytywania RAW"""
    fname = os.path.basename(path)
    match = re.search(r'[x_](\d+)x(\d+)', fname)
    if match:
        w, h = int(match.group(1)), int(match.group(2))
    else:
        w, h = 512, 512
    try:
        img = np.fromfile(path, dtype=np.float32).reshape((h, w))
    except ValueError:
        img = np.fromfile(path, dtype=np.uint16).astype(np.float32).reshape((h, w))
    return img, (h, w)

def read_png_file(path):
    """Logika wczytywania PNG"""
    # Wczytanie jako grayscale (L)
    img_pil = Image.open(path).convert('L')
    img = np.array(img_pil).astype(np.float32)
    # PNG jest w zakresie 0-255, więc normalizujemy do 0-1 tymczasowo (reszta logiki jest wspólna)
    img = img / 255.0
    h, w = img.shape
    return img, (h, w)

def read_image(path, file_format="png", target_size=(512, 512), normalize=True):
    """
    Uniwersalna funkcja wczytująca (RAW lub PNG) i zwracająca obraz w zakresie modelu [-1, 1]
    """
    if file_format == "raw":
        img, (h, w) = read_raw_file(path, target_size)
    else:
        img, (h, w) = read_png_file(path) # Tu img jest już float 0-1 (z 0-255) lub float (jeśli raw)

    # Normalizacja do zakresu [-1, 1] (wymagane przez generator Tanh)
    # Dla RAW musimy zrobić Min-Max, bo wartości mogą być dowolne (np. Hounsfield)
    # Dla PNG (które już podzieliliśmy przez 255) też robimy min-max, żeby rozciągnąć histogram
    if normalize:
        min_v, max_v = img.min(), img.max()
        if max_v - min_v > 1e-6:
            img = (img - min_v) / (max_v - min_v) # [0, 1]
            img = img * 2.0 - 1.0 # [-1, 1]
        else:
            img = np.zeros_like(img)
            
    if (h, w) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
    return img


def get_advanced_guidance(img_tensor, max_metals=2, irregular_shapes=False):
    """
    Tworzy prowadnice dla wielu implantów o opcjonalnie nieregularnych kształtach,
    przycięte do sylwetki pacjenta.
    """
    img = img_tensor
    h, w = img.shape
    
    # Detekcja sylwetki pacjenta
    body_limit = (img > -0.9).astype(np.float32)
    
    guidance = np.zeros((h, w), dtype=np.float32)
    mask_metal = np.zeros((h, w), dtype=np.uint8)
    
    # Losujemy liczbę implantów (od 1 do max_metals)
    num_metals = np.random.randint(1, max_metals + 1)
    
    for _ in range(num_metals):
        # Losowanie centrum implantu wewnątrz obszaru pacjenta
        center = (np.random.randint(180, 332), np.random.randint(180, 332))
        
        if irregular_shapes:
            # Tworzenie nieregularnego wielokąta
            num_pts = np.random.randint(4, 9) # Liczba wierzchołków
            pts = []
            for i in range(num_pts):
                # Losujemy punkty w promieniu 5-15 pikseli od centrum
                dist = np.random.randint(5, 15)
                angle = (i / num_pts) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
                px = int(center[0] + dist * np.cos(angle))
                py = int(center[1] + dist * np.sin(angle))
                pts.append([px, py])
            
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask_metal, [pts], 255)
        else:
            # Klasyczne kółko
            radius = np.random.randint(3, 9)
            cv2.circle(mask_metal, center, radius, 255, -1)
        
        # Generowanie smug dla konkretnego centrum
        num_streaks = np.random.randint(5, 50)
        for _ in range(num_streaks):
            angle = np.random.uniform(0, 360)
            rad = np.deg2rad(angle)
            length = 800
            thickness = np.random.choice([1, 3], p=[0.9, 0.1])
            intensity = np.random.uniform(0.3, 0.7)
            
            end_x = int(center[0] + length * np.cos(rad))
            end_y = int(center[1] + length * np.sin(rad))
            cv2.line(guidance, center, (end_x, end_y), intensity, thickness, cv2.LINE_AA)

    # Przycinanie i post-processing
    guidance = guidance * body_limit
    guidance = cv2.GaussianBlur(guidance, (3, 3), 0)
    
    # Finalne złożenie obrazów
    mask_float = mask_metal / 255.0
    img_with_metal = np.clip(img + mask_float, -1.0, 1.0)
    # Prowadnica (guidance) + metal dla wyraźnego zaznaczenia źródła
    final_g = np.clip(guidance + mask_float, 0.0, 1.0)
    
    return torch.from_numpy(img_with_metal).unsqueeze(0).float(), torch.from_numpy(final_g).unsqueeze(0).float(), torch.from_numpy(mask_float).unsqueeze(0).float()

def tensor_to_01(t):
    return (t * 0.5) + 0.5

def calculate_roi_metrics(gen, real, mask):
    gen_sq = gen.squeeze()
    real_sq = real.squeeze()
    mask_sq = mask.squeeze()
    
    # 1. Metryki Tła
    bg_mask = 1.0 - mask_sq
    bg_pixel_count = torch.sum(bg_mask).item()
    
    bg_psnr = float('nan')
    bg_mae = float('nan')
    
    if bg_pixel_count > 0:
        mse_bg = torch.sum((gen_sq - real_sq)**2 * bg_mask) / bg_pixel_count
        if mse_bg > 0:
            bg_psnr = 10 * torch.log10(1.0 / mse_bg).item()
        bg_mae = torch.sum(torch.abs(gen_sq - real_sq) * bg_mask) / bg_pixel_count

    # 2. Metryki Artefaktu
    mask_np = mask_sq.cpu().numpy().astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    
    art_mask = torch.from_numpy(dilated_mask_np).to(gen.device).float()
    art_pixel_count = torch.sum(art_mask).item()
    
    art_mae = float('nan')
    art_mse = float('nan')
    
    if art_pixel_count > 0:
        diff = (gen_sq - real_sq) * art_mask
        art_mae = (torch.sum(torch.abs(diff)) / art_pixel_count).item()
        art_mse = (torch.sum(diff**2) / art_pixel_count).item()

    return {
        "Bg_PSNR": bg_psnr,
        "Bg_MAE": bg_mae.item() if isinstance(bg_mae, torch.Tensor) else bg_mae,
        "Art_MAE": art_mae,
        "Art_MSE": art_mse
    }

def frequency_energy(img):
    fft = torch.fft.fft2(img.squeeze())
    return torch.mean(torch.abs(fft)).item()

def evaluate():
    print(f"--- START EWALUACJI ---")
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Dane testowe: {TEST_DIR}")
    print(f"Format wejściowy: {INPUT_FORMAT.upper()}")
    
    # 1. Inicjalizacja metryki stylu
    try:
        style_metric = StyleLossMetric(DEVICE)
    except Exception as e:
        print(f"OSTRZEŻENIE: Błąd StyleLoss: {e}")
        style_metric = None

    # 2. Ładowanie modelu
    if MODEL_TYPE == "standard":
        model = GenADN_Final().to(DEVICE)
    elif MODEL_TYPE == "vae":
        model = GenADN_VAE().to(DEVICE)
    else:
        raise ValueError("Nieznany typ modelu")
        
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"BŁĄD: Brak pliku wagi: {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model załadowany.")

    # 3. Zbieranie plików
    for i in range(1, 4):
        dir_target = os.path.join(TEST_DIR_PARAMETRIZED(i), "clean")
        dir_art = os.path.join(TEST_DIR_PARAMETRIZED(i), "artifact")
        # dir_mask = os.path.join(TEST_DIR, "Mask")
        
        # Wybór rozszerzenia na podstawie flagi
        ext = "*.raw" if INPUT_FORMAT == "raw" else "*.png"
        
        f_targets = sorted(glob.glob(os.path.join(dir_target, ext)))
        f_arts = sorted(glob.glob(os.path.join(dir_art, ext)))
        # f_masks = sorted(glob.glob(os.path.join(dir_mask, ext)))
        
        min_len = min(len(f_targets), len(f_arts))
        print(f"Znaleziono {min_len} kompletnych trójek testowych ({ext}).")
        
        results = []

        with torch.no_grad():
            for i in tqdm(range(min_len), desc="Przetwarzanie"):
                # A. Wczytywanie danych (używamy nowej funkcji read_image)
                y_np = read_image(f_targets[i], file_format=INPUT_FORMAT)
                x_a_np = read_image(f_arts[i], file_format=INPUT_FORMAT)
                y_m, g, m = get_advanced_guidance(y_np, max_metals=3, irregular_shapes=True)
                y_m, g, m = y_m.unsqueeze(0).to(DEVICE), g.unsqueeze(0).to(DEVICE), m.unsqueeze(0).to(DEVICE)
                
                # Maska: wczytujemy bez normalizacji [-1,1], ale musimy ją zrzutować na 0-1
                # read_image zwraca [-1, 1], więc konwertujemy ręcznie
                # m_np_norm = read_image(f_masks[i], file_format=INPUT_FORMAT, normalize=True)
                # # Thresholding maski (wszystko powyżej tła to maska)
                # m_np = (m_np_norm > -0.9).astype(np.float32) 

                # Przygotowanie tensorów
                y = torch.from_numpy(y_np).float().view(1,1,512,512).to(DEVICE)     # Target
                x_a = torch.from_numpy(x_a_np).float().view(1,1,512,512).to(DEVICE) # Baseline
                # m = torch.from_numpy(m_np).float().view(1,1,512,512).to(DEVICE)     # Mask
                
                # g_np = get_guidance_from_mask(m_np)
                # g = torch.from_numpy(g_np).float().view(1,1,512,512).to(DEVICE)

                # B. Generowanie
                if MODEL_TYPE == "standard":
                    z = model.EA(x_a)
                    fake_art = model.GA(model.EC(y, g), z)
                else:
                    z, mu, logvar = model.EA(x_a)
                    fake_art = model.GA(model.EC(y, g), z)

                if INFERENCE_ONLY:
                    for image in fake_art:
                        image_number = os.path.basename(f_targets[i]).split('.')[0]
                        save_image(image * 0.5 + 0.5, f"{OUTPUT_DIR}/{image_number}_gen.png")
                        # print(f"Zapisano wygenerowany obraz: {OUTPUT_DIR}/{image_number}_gen.png")
                    continue
                # C. Obliczanie metryk
                gen_01 = tensor_to_01(fake_art)
                real_01 = tensor_to_01(y)
                art_01 = tensor_to_01(x_a)
                gen_01 = torch.clamp(gen_01, 0, 1)
                art_01 = torch.clamp(art_01, 0, 1)
                real_01 = torch.clamp(real_01, 0, 1)

                val_psnr = psnr(gen_01, real_01, data_range=1.0).item()
                val_ssim = ssim(gen_01, real_01, data_range=1.0).item()
                val_fsim = fsim(gen_01, real_01, data_range=1.0, chromatic=False).item()
                val_mae_global = torch.mean(torch.abs(gen_01 - real_01)).item()
                
                val_style = float('nan')
                if style_metric is not None:
                    val_style = style_metric(gen_01, art_01)

                roi_metrics = calculate_roi_metrics(gen_01, real_01, m)

                freq_gen = frequency_energy(gen_01)
                freq_real = frequency_energy(art_01)

                # TODO: Dodac LPIPS
                # import lpips
                # lpips_metric = lpips.LPIPS(net='alex').to(DEVICE)
                # lpips = lpips_metric(fake_art.repeat(1,3,1,1), x_a.repeat(1,3,1,1)).mean().item()

                results.append({
                    "Filename": os.path.basename(f_targets[i]),
                    "Global_PSNR": val_psnr,
                    "Global_SSIM": val_ssim,
                    "Global_FSIM": val_fsim,
                    "StyleLoss_gen_to_art": val_style,
                    "Global_MAE": val_mae_global,
                    "Anatomy_PSNR": roi_metrics["Bg_PSNR"],
                    "Anatomy_MAE": roi_metrics["Bg_MAE"],
                    "Artifact_MAE": roi_metrics["Art_MAE"],
                    "Artifact_MSE": roi_metrics["Art_MSE"],
                    "Freq_Generated": freq_gen,
                    "Freq_Real": freq_real,
                    # "LPIPS_Realism": val_lpips
                })

        if not INFERENCE_ONLY:
            # Podsumowanie i zapis
            df = pd.DataFrame(results)
            print("\n--- WYNIKI ŚREDNIE ---")
            print(df.mean(numeric_only=True))
            
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nPełne wyniki zapisano w: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate()