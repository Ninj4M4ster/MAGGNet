import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import os, cv2, numpy as np, itertools, glob
from PIL import Image
from tqdm import tqdm
from gen_adn_advanced import GenADN_Final

# --- KONFIGURACJA ---
DIR_FREE = "data/train_free"
DIR_ART  = "data/train_artifact"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-4
LAMBDA_CONTENT = 30.0    # Silna ochrona anatomii
LAMBDA_EDGE = 1.5       # Wyrazistość smug
LAMBDA_BG = 50.0        # Ekstremalna kara za brud w tle
LAMBDA_METAL = 15.0

def get_advanced_guidance(img_tensor, max_metals=3, irregular_shapes=False):
    """
    Tworzy prowadnice dla wielu implantów o opcjonalnie nieregularnych kształtach,
    przycięte do sylwetki pacjenta.
    """
    img = img_tensor.squeeze().cpu().numpy()
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
        num_streaks = np.random.randint(20, 100)
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
    
    return torch.from_numpy(img_with_metal).unsqueeze(0).float(), \
           torch.from_numpy(final_g).unsqueeze(0).float(), \
           torch.from_numpy(mask_float).unsqueeze(0).float()

def train():
    os.makedirs("samples_advanced_multipleguide", exist_ok=True)
    model = GenADN_Final().to(DEVICE)
    opt_G = optim.Adam(itertools.chain(model.EC.parameters(), model.EA.parameters(), model.GA.parameters()), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(model.DA.parameters(), lr=LR, betas=(0.5, 0.999))
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=DEVICE).float().view(1,1,3,3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=DEVICE).float().view(1,1,3,3)

    files_f = sorted(glob.glob(os.path.join(DIR_FREE, "*.*")))
    files_a = sorted(glob.glob(os.path.join(DIR_ART, "*.*")))

    for epoch in range(100):
        np.random.shuffle(files_f)
        np.random.shuffle(files_a)
        
        loop = tqdm(range(min(len(files_f), len(files_a))), desc=f"Epoch {epoch}")
        for i in loop:
            y = torch.from_numpy(np.array(Image.open(files_f[i]).convert('L').resize((512,512)))/127.5-1).float().view(1,1,512,512).to(DEVICE)
            x_a = torch.from_numpy(np.array(Image.open(files_a[i]).convert('L').resize((512,512)))/127.5-1).float().view(1,1,512,512).to(DEVICE)

            y_m, g, m = get_advanced_guidance(y[0])
            y_m, g, m = y_m.unsqueeze(0).to(DEVICE), g.unsqueeze(0).to(DEVICE), m.unsqueeze(0).to(DEVICE)

            # --- G STEP ---
            opt_G.zero_grad()
            fake = model.GA(model.EC(y_m, g), model.EA(x_a))

            # KLUCZ 2: Maska ciała dla tła
            body_mask = (y > -0.9).float()
            
            # Wymuszamy czyste tło w samym obrazie wygenerowanym przed stratami
            fake_final = fake * body_mask + y * (1 - body_mask)

            # 1. GAN Loss
            loss_GAN = F.mse_loss(model.DA(fake_final), torch.ones_like(model.DA(fake_final)))

            # 2. Content Loss (Chroni anatomię tam gdzie g jest słabe)
            content_mask = (body_mask * (g < 0.2)).float()
            loss_content = F.l1_loss(fake_final * content_mask, y * content_mask) * LAMBDA_CONTENT

            # 3. Background Loss (Ekstremalne czyszczenie tła)
            loss_bg = F.mse_loss(fake * (1 - body_mask), y * (1 - body_mask)) * LAMBDA_BG

            # 4. Edge Loss (Tylko wewnątrz ciała, tam gdzie jest g)
            gx = F.conv2d(fake_final, sobel_x, padding=1)
            gy = F.conv2d(fake_final, sobel_y, padding=1)
            loss_edge = -torch.mean(torch.sqrt(gx**2 + gy**2 + 1e-8) * (g > 0.1).float()) * LAMBDA_EDGE

            total_G = loss_GAN + loss_content + loss_edge + loss_bg + (F.mse_loss(fake_final * m, m * 1.0) * LAMBDA_METAL)
            total_G.backward()
            opt_G.step()

            # --- D STEP ---
            opt_D.zero_grad()
            loss_D = (F.mse_loss(model.DA(x_a), torch.ones_like(model.DA(x_a))) + 
                      F.mse_loss(model.DA(fake_final.detach()), torch.zeros_like(model.DA(fake_final)))) * 0.5
            loss_D.backward()
            opt_D.step()

            if i % 100 == 0:
                loop.set_postfix(G=total_G.item(), D=loss_D.item(), Edge=loss_edge.item())
                # Preview: [Clean | Guidance | Style | Result]
                preview = torch.cat([y, g*2-1, x_a, fake_final], dim=3)
                save_image(preview * 0.5 + 0.5, f"samples_advanced_multipleguide/e{epoch}_{i}.png")

        torch.save(model.state_dict(), f"checkpoints/gen_adn_final_multipleguide.pth")

if __name__ == "__main__":
    print("A")
    # train()