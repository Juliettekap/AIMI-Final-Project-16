import torch
import os
import argparse

def process_model_weights(input_ckpt_path, output_ckpt_path):
    '''
    Loads model checkpoint that was stored at train-time, discards
    "optimizer_state_dict" and "epoch", and only keeps the trained 
    model weights (i.e. "model_state_dict"). Reduces memory footprint
    of weights file by nearly 5x.
    '''
    checkpoint = torch.load(input_ckpt_path, map_location=torch.device('cpu'))
    torch.save({
        'model_state_dict': checkpoint['model_state_dict']}, output_ckpt_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim model size.")
    parser.add_argument("folder_path", help="The path to the folder containing model files.")

    args = parser.parse_args()

    for filename in os.listdir(args.folder_path):
        if filename.endswith('.pt'):
            process_model_weights(args.folder_path + '/' + filename, args.folder_path + '/' + filename)