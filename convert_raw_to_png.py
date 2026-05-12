import os
import numpy as np
import cv2  # OpenCV handles 16-bit PNGs better than PIL sometimes
import argparse
from tqdm import tqdm
from glob import glob

def save_as_8bit_png(img_array, output_path, wl=40, ww=400):
    """
    Konwertuje tablicę HU (img_array) na widoczny 8-bitowy obraz PNG (0-255) 
    z zastosowaniem okienkowania CT.
    
    Args:
        img_array (np.ndarray): Tablica wartości HU (np. float32, int16).
        output_path (str): Ścieżka zapisu pliku PNG.
        wl (float): Window Level (Centrum okna HU).
        ww (float): Window Width (Szerokość okna HU).
    """
    
    # 1. Określenie granic okna
    window_min = wl - (ww / 2)
    window_max = wl + (ww / 2)
    
    # 2. Mapowanie jednostek HU na zakres [0, 255]
    
    # a) Ograniczenie wartości HU do zakresu okna (Clamping)
    img_windowed = np.clip(img_array, window_min, window_max)
    
    # b) Normalizacja do [0, 1]
    # Wartość 0 jest na window_min, wartość 1 jest na window_max
    img_normalized = (img_windowed - window_min) / ww
    
    # c) Skalowanie do [0, 255] i konwersja na typ integer
    # Wartości są ograniczane do 0-255 i zapisywane jako uint8.
    img_8bit = (img_normalized * 255.0).astype(np.uint8)
    
    # 3. Zapis używając OpenCV
    cv2.imwrite(output_path, img_8bit)

def process_raw_dataset(source_dir, output_dir, width=512, height=512):
    # Output structure
    clean_dir = os.path.join(output_dir, "clean")
    artifact_dir = os.path.join(output_dir, "artifact")
    
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"Searching for .raw files in: {source_dir}")
    raw_files = glob(os.path.join(source_dir, "**", "*.raw"), recursive=True)
    
    if not raw_files:
        print("No .raw files found!")
        return

    # Dictionary for pairing: ID -> {'clean': path, 'artifact': path}
    pairs = {}

    print("Indexing files...")
    for fpath in raw_files:
        fname = os.path.basename(fpath)
        
        # Identify type
        is_clean = "nometal" in fname and not 'segmentation' in fname
        is_artifact = "metalart" in fname and not 'segmentation' in fname
        
        if not (is_clean or is_artifact):
            continue
            
        # Extract ID (assuming format ..._imgX_...)
        try:
            parts = fname.split('_')
            # Look for part starting with "img"
            img_id_part = next((p for p in parts if p.startswith("img")), None)
            
            if img_id_part:
                # Add prefix "head" or "body" to ensure uniqueness
                body_part = "head" if "head" in fname else "body"
                unique_id = f"{body_part}_{img_id_part}"
            else:
                unique_id = fname
                
        except Exception:
            print(f"Skipping file with weird name: {fname}")
            continue

        if unique_id not in pairs:
            pairs[unique_id] = {}
            
        if is_clean:
            pairs[unique_id]['clean'] = fpath
        else:
            pairs[unique_id]['artifact'] = fpath

    print(f"Found {len(pairs)} potential pairs. Processing...")
    
    processed_count = 0
    
    for pid, paths in tqdm(pairs.items()):
        if 'clean' in paths and 'artifact' in paths:
            # We have a complete pair
            path_clean = paths['clean']
            path_artifact = paths['artifact']
            
            try:
                # Read RAW data
                # Assumption: Medical data is often float32.
                # If your raw data is different (e.g. int16), change dtype here.
                # Based on file snippets (lots of 'Ä'), it looks like binary float data.
                arr_clean = np.fromfile(path_clean, dtype=np.float32).reshape((height, width))
                arr_art = np.fromfile(path_artifact, dtype=np.float32).reshape((height, width))
                
            except Exception as e:
                print(f"Read error {pid}: {e}")
                continue
                
            # Save as 16-bit PNG (preserving HU values via offset)
            out_name = f"{pid}.png"
            
            save_as_8bit_png(arr_clean, os.path.join(clean_dir, out_name))
            save_as_8bit_png(arr_art, os.path.join(artifact_dir, out_name))
            
            processed_count += 1
            
    print(f"\nDone! Processed {processed_count} pairs.")
    print(f"Saved in: {output_dir}")
    print("NOTE: PNGs are saved as 16-bit uint16 with +32768 offset.")
    print("To get real HU in your training loop, use: image_hu = image_loaded - 32768")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data_rpi", help="Folder with .raw files")
    parser.add_argument("--output", default="dataset_adn_hu", help="Output folder")
    args = parser.parse_args()
    
    process_raw_dataset(args.source, args.output)