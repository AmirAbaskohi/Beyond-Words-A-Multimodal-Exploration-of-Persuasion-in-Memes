from utils import (
    download_image_object, check_addresses_exist,
    create_necessary_directories, read_dataset_file,
    concatenate_captions
)
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import argparse
from datasets import Image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process meme caption dataset.")
    
    parser.add_argument('--dataset_path', type=str, help='Path to meme caption dataset')
    parser.add_argument('--save_address', type=str, help='Address to save the data')

    args = parser.parse_args()

    dataset_path = args.dataset_path or './memecap/memes-trainval.json'
    save_address = args.save_address or './'
    
    check_addresses_exist(save_address, dataset_path)
    blip_dataset_path, images_path = create_necessary_directories(save_address, 'blip_dataset')

    meme_cap_dataset = read_dataset_file(dataset_path)
    
    images = []
    image_captions = []

    for d in tqdm(meme_cap_dataset):
        image = None

        try:
            image = download_image_object(d['url'])
        except:
            print("This data is skipped because of an error when downloading the image.")
            continue

        image_caption = concatenate_captions(d['img_captions'], d['meme_captions'])
        
        images.append(image)
        image_captions.append(image_caption)

    df = pd.DataFrame({"image": images, "text": image_captions})
    dataset = Dataset.from_pandas(df)

    print("Saving llava dataset ...")
    dataset.save_to_disk(blip_dataset_path)


