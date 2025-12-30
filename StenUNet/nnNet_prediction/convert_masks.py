import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def convert_masks(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    files = glob(os.path.join(input_dir, '*.png'))
    print(f"Found {len(files)} images in {input_dir}")
    
    count = 0
    for f in tqdm(files):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read {f}")
            continue
            
        # Convert 0-1 to 0-255
        img_255 = (img * 255).astype(np.uint8)
        
        # Save to output dir
        basename = os.path.basename(f)
        out_path = os.path.join(output_dir, basename)
        cv2.imwrite(out_path, img_255)
        count += 1
            
    print(f"Converted {count} images. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 0-1 masks to 0-255 masks")
    parser.add_argument("--input", "-i", type=str, default="StenUNet/nnNet_prediction/Dataset001_ARCADEseg", help="Input directory containing .png masks")
    parser.add_argument("--output", "-o", type=str, default="StenUNet/nnNet_prediction/Dataset001_ARCADEseg_255", help="Output directory to save converted masks")
    
    args = parser.parse_args()
    
    convert_masks(args.input, args.output)
