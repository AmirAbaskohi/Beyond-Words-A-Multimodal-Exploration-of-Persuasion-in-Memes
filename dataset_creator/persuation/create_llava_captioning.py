from utils import (
    check_addresses_exist,
    create_necessary_directories, read_dataset_file
)
from tqdm import tqdm
import argparse
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process meme caption dataset.")
    
    parser.add_argument('--annotation_address', type=str, help='Path to annotation file')
    parser.add_argument('--output_location', type=str, help='Path to the output folder')

    args = parser.parse_args()

    annotation_address = args.annotation_address or './data/subtask2a/validation.json'
    output_location = args.output_location or './'
    
    check_addresses_exist(annotation_address, output_location)
    llava_dataset_path = create_necessary_directories(output_location, 'pesuation_captioning_llava_dataset')

    persuation_dataset = read_dataset_file(annotation_address)
    llava_captioning_dataset = []

    question_id = 0
    for d in tqdm(persuation_dataset):   
        text = "This is a meme with the following text written inside the meme: \n \""
        text += d["text"].replace("\\n", "\n").strip("\n").replace("\n", " \n ").strip()
        text += "\". \n What is the meme poster trying to convey?"

        llava_captioning_dataset.append({
            "image": d["image"],
            "text": text,
            "question_id": question_id
        })

        question_id += 1

    print("Saving llava dataset ...")
    with open(os.path.join(llava_dataset_path, 'infer.jsonl'), 'w') as file:
        for obj in llava_captioning_dataset:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')