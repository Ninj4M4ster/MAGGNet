import numpy as np
from PIL import Image
import os
import glob
import argparse
import sys

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

def convert_16bit_to_8bit(file_path, output_dir, window_level=None, window_width=None):
    """
    Reads a 16-bit PNG, applies windowing/normalization, and saves as 8-bit PNG.
    """
    try:
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

        # 5. Save
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, base_name) # Keep same name
        
        output_img.save(output_path)
        print(f"[OK] Converted: {base_name}")

    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}", file=sys.stderr)

def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Convert 16-bit Medical PNGs (HU) to standard 8-bit PNGs.")
    
    parser.add_argument("--input", type=str, default=".", help="Path to input folder with 16-bit PNGs.")
    parser.add_argument("--output", type=str, default="converted_8bit", help="Path to output folder.")
    
    # Windowing arguments
    parser.add_argument("--level", type=float, default=None, help="Window Center (Level). Recommended for medical visibility.")
    parser.add_argument("--width", type=float, default=None, help="Window Width.")

    args = parser.parse_args()

    # --- Execution ---
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    # Find PNG files
    search_pattern = os.path.join(args.input, "**", "*.png")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"No PNG files found in: {args.input}")
        return

    print(f"Found {len(files)} images. Processing...")
    
    if args.level is not None:
        print(f"Mode: Medical Windowing (L={args.level}, W={args.width})")
    else:
        print("Mode: Full Range Auto-scale (Min-Max)")

    for f in files:
        # Avoid processing the output folder if it's inside input
        if os.path.abspath(args.output) in os.path.abspath(f):
            continue
            
        convert_16bit_to_8bit(f, args.output, args.level, args.width)

    print("\nJob done.")

if __name__ == "__main__":
    main()