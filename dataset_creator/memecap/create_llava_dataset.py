from utils import (
    download_image, check_addresses_exist,
    create_necessary_directories, read_dataset_file,
    concatenate_captions
)
from tqdm import tqdm
import argparse
import json
import uuid
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process meme caption dataset.")
    
    parser.add_argument('--dataset_path', type=str, help='Path to meme caption dataset')
    parser.add_argument('--save_address', type=str, help='Address to save the data')

    args = parser.parse_args()

    dataset_path = args.dataset_path or './memecap/memes-trainval.json'
    save_address = args.save_address or './'
    
    check_addresses_exist(save_address, dataset_path)
    llava_dataset_path, images_path = create_necessary_directories(save_address, 'llava_dataset')

    meme_cap_dataset = read_dataset_file(dataset_path)
    llava_dataset = []

    for d in tqdm(meme_cap_dataset):
        generated_id = str(uuid.uuid4())
        
        try:
            download_image(d['url'], os.path.join(images_path, f"{generated_id}.jpg"))
        except:
            print("This data is skipped because of an error when downloading the image.")
            continue
        
        image_caption = concatenate_captions(d['img_captions'], d['meme_captions'])

        llava_dataset.append({
            "id": generated_id,
            "image": f"{generated_id}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat this meme is trying to convey?"
                },
                {
                    "from": "gpt",
                    "value": image_caption
                }
            ]
        })

    print("Saving llava dataset ...")
    with open(os.path.join(llava_dataset_path, 'llava_dataset.json'), 'w') as llava_dataset_file:
        json.dump(llava_dataset, llava_dataset_file, indent=4)