import argparse
import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Change llava dataset from json to jsonl for inference.")
    
    parser.add_argument('--path', type=str, help='Path to llava json dataset file.')

    args = parser.parse_args()
    
    file_path = args.path

    with open(file_path, 'r') as file:
        data = json.load(file)

    directory, original_file = os.path.split(file_path)
    new_file = "infer_ocr.jsonl"
    new_path = os.path.join(directory, new_file)

    question_id = 0
    with open(new_path, 'w') as file:
        for obj in tqdm(data):
            infer_obj = { 
                "image": obj["image"],
                "text": obj["conversations"][0]["value"][7:],
                "question_id": question_id
            }
            json_line = json.dumps(infer_obj)
            file.write(json_line + '\n')
            
            question_id += 1