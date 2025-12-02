import argparse
import os
from pathlib import Path
import torch

from utils.util import *

from pre_process.preprocess import preprocess
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data as predict
from post_process.remove_small_segments import remove_small_segments

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=False, default='./model_folder', help = 'model folder')
    parser.add_argument('-chk', type=str, required=True, default='./model_folder/model_final.pth',help = 'checkpoint name, put in the model folder')
    
    parser.add_argument('-i', type=str, required=False, default='./dataset_test/raw/', help = 'Folder in which the raw test images are')
    parser.add_argument('-o', type=str, required=False, default='./dataset_test/raw_prediction/', help = 'Folder in which the raw predictions are')
    parser.add_argument('-p', type=str, required=False, default='./dataset_test/post_prediction/', help = 'Folder in which the post predictions are')
    
    parser.add_argument('-t', type=int, required=False, default=600, help = 'Threshold to remove small segments')
    parser.add_argument('--use-pseudo-color-map', action='store_true', help='Append predicted pseudo color channels during inference.')
    parser.add_argument('--pseudo-color-dir', type=str, default='StenUNet/pseudo_color/runs/inferences',
                        help='Root directory of predicted pseudo color outputs.')
    parser.add_argument('--no-pseudo-color-normalize', action='store_false', dest='pseudo_color_normalize',
                        help='Disable normalization of pseudo color channels before concatenation.')
    parser.set_defaults(pseudo_color_normalize=True)
 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_pseudo_color_map:
        os.environ["STENUNET_USE_PSEUDO_COLOR_MAP"] = "1"
        os.environ["STENUNET_PSEUDO_COLOR_DIR"] = args.pseudo_color_dir
        os.environ["STENUNET_PSEUDO_COLOR_NORMALIZE"] = "1" if args.pseudo_color_normalize else "0"
    else:
        os.environ.pop("STENUNET_USE_PSEUDO_COLOR_MAP", None)
        os.environ.pop("STENUNET_PSEUDO_COLOR_DIR", None)
        os.environ.pop("STENUNET_PSEUDO_COLOR_NORMALIZE", None)

    print('--------------preprocessing--------------')

    input_path = Path(args.i).expanduser()
    preprocess_path = input_path.parent / "preprocessed"
    mkdir(str(preprocess_path))
    
    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue
        img_array = image_to_array(str(file_path))
        pre_img = preprocess(img_array)
        cv2.imwrite(str(preprocess_path / file_path.name), pre_img)
    print('--------------preprocessing done--------------')
    mkdir(args.o)
    print('--------------predicting--------------')
    predict(list_of_lists_or_source_folder = str(preprocess_path), output_folder = args.o, model_training_output_dir = args.m, use_folds =[0], checkpoint_name=args.chk, num_processes_preprocessing=1, num_processes_segmentation_export=1,device = device)
    print('--------------postprocessing--------------')
    mkdir(args.p)
    remove_small_segments(args.o, args.p, threshold = args.t)
    print('--------------Done--------------')
    
