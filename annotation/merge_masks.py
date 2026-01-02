import cv2
import numpy as np
import os
from tqdm import tqdm

def main():
    pred_dir = '/home/data/nas_hdd/dazhou/centerline/StenUNet/nnNet_prediction/Dataset001_ARCADEseg_train_255'
    gt_dir = '/home/data/nas_hdd/dazhou/centerline/annotation/labelsTr'
    out_dir = '/home/data/nas_hdd/dazhou/centerline/annotation/labelsTr_OR'

    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    print(f"Found {len(files)} files in {pred_dir}")

    for f in tqdm(files):
        pred_path = os.path.join(pred_dir, f)
        gt_path = os.path.join(gt_dir, f)
        out_path = os.path.join(out_dir, f)

        if not os.path.exists(gt_path):
            print(f"Warning: {f} not found in {gt_dir}, skipping.")
            continue

        img_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img_pred is None:
            print(f"Error reading {pred_path}")
            continue
        if img_gt is None:
            print(f"Error reading {gt_path}")
            continue

        # Bitwise OR
        img_or = cv2.bitwise_or(img_pred, img_gt)

        cv2.imwrite(out_path, img_or)

    print("Done.")

if __name__ == "__main__":
    main()
