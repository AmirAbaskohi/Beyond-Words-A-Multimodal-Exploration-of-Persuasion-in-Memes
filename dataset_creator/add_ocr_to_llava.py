import argparse
import json
import os
from tqdm import tqdm
import copy
import easyocr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add OCR to the lava dataset created.")
    
    parser.add_argument('--data_path', type=str, help='Path to llava json dataset file.')
    parser.add_argument('--images_path', type=str, help='Path to the llava dataset\'s images exist.')

    args = parser.parse_args()

    with open(args.data_path, 'r') as file:
        data = json.load(file)

    directory, original_file = os.path.split(args.data_path)
    new_file = "llava_dataset_ocr.json"
    new_path = os.path.join(directory, new_file)

    llava_dataset_ocr = []

    reader = easyocr.Reader(['en'])

    for obj in tqdm(data):
        copy_obj = copy.deepcopy(obj)
        
        new_value = ""
        new_value += "<image>This is a meme with the following text written inside the meme: \n"
        new_value += "\""
        new_value += "\n".join(reader.readtext(os.path.join(args.images_path, obj['image']), detail=0))
        new_value += "\". \n"
        new_value += "What is the meme poster trying to convey?"
        
        copy_obj['conversations'][0]['value'] = new_value

        llava_dataset_ocr.append(copy_obj)

    with open(new_path, 'w') as llava_dataset_file:
        json.dump(llava_dataset_ocr, llava_dataset_file, indent=4)