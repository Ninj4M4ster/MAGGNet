import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import shutil

# --- GLOBALNE STAŁE ---
IMAGE_SIZE = 512
IN_CHANNELS = 1

# --- STANDARDOWA TRANSFORMACJA CT ---
# Skalowanie, konwersja do tensora i normalizacja do zakresu [-1, 1]
CT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # Normalizacja do zakresu [-1, 1], ponieważ dekodery używają Tanh
    transforms.Normalize((0.5,), (0.5,)) 
])

def load_image(path):
    """Ładuje i przetwarza pojedynczy obraz PNG w skali szarości."""
    img = Image.open(path).convert('L') # 'L' dla skali szarości
    return CT_TRANSFORM(img)

# --- FUNKCJA PODZIAŁU DANYCH ---

def split_data_to_folders(art_dir, free_dir, train_art_dest, train_free_dest, test_art_dest, test_free_dest, test_split):
    """
    Dzieli sparowane dane z master folderów na zbiory treningowe (unpaired) 
    i testowe (paired) w folderach roboczych.
    """
    
    # Utwórz foldery robocze
    for d in [train_art_dest, train_free_dest, test_art_dest, test_free_dest]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Pobierz sparowane nazwy plików i wymieszaj
    all_files = sorted([f for f in os.listdir(art_dir) if f.endswith('.png')])
    random.seed(42) # Ustawienie seeda dla powtarzalności podziału
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * test_split)
    test_files = all_files[:split_idx]
    train_files = all_files[split_idx:]
    
    print(f"Dane: {len(all_files)} plików. Podział: Trening ({len(train_files)}), Test ({len(test_files)})")

    # Kopiowanie plików do folderów roboczych
    for filename in train_files:
        # Kopiowanie obrazów z artefaktami i bez
        shutil.copy(os.path.join(art_dir, filename), os.path.join(train_art_dest, filename))
        shutil.copy(os.path.join(free_dir, filename), os.path.join(train_free_dest, filename))

    for filename in test_files:
        # Kopiowanie sparowanych obrazów do folderów testowych
        shutil.copy(os.path.join(art_dir, filename), os.path.join(test_art_dest, filename))
        shutil.copy(os.path.join(free_dir, filename), os.path.join(test_free_dest, filename))

    print("Podział danych zakończony.")
    return len(train_files), len(test_files)


# --- KLASY DATASET ---

class UnpairedTrainDataset(Dataset):
    """
    Zbiór danych treningowych dla ADN (Nienadzorowany/Unpaired).
    Zapewnia parę obrazów (x_a, y) losowo dobraną z dwóch domen.
    """
    def __init__(self, artifact_dir, free_dir):
        self.artifact_files = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith('.png')]
        self.free_files = [os.path.join(free_dir, f) for f in os.listdir(free_dir) if f.endswith('.png')]
        
        # Długość datasetu to max. liczba plików, aby zapewnić, że wszystkie zostaną użyte
        self.length = max(len(self.artifact_files), len(self.free_files))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 1. Obraz z artefaktem (x_a) - Cyklicznie
        artifact_path = self.artifact_files[index % len(self.artifact_files)]
        x_a = load_image(artifact_path)
        
        # 2. Obraz bez artefaktu (y) - Losowo z drugiej domeny
        random_idx = random.randint(0, len(self.free_files) - 1)
        free_path = self.free_files[random_idx]
        y = load_image(free_path)
        
        return x_a, y

class PairedTestDataset(Dataset):
    """
    Zbiór danych testowych (Nadzorowany/Paired).
    Zapewnia sparowaną parę (x_a, GT).
    """
    def __init__(self, artifact_dir, free_dir):
        artifact_filenames = sorted([f for f in os.listdir(artifact_dir) if f.endswith('.png')])
        
        self.paired_files = []
        for filename in artifact_filenames:
            artifact_path = os.path.join(artifact_dir, filename)
            free_path = os.path.join(free_dir, filename)
            
            if os.path.exists(free_path):
                self.paired_files.append((artifact_path, free_path))

        self.length = len(self.paired_files)
        print(f"Załadowano {self.length} sparowanych par do testu.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        artifact_path, free_path = self.paired_files[index]
        
        x_a = load_image(artifact_path)
        gt = load_image(free_path) # Ground Truth
        
        return x_a, gt # x_a (wejście), gt (prawda gruntowa)