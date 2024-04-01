from io import BytesIO
from PIL import Image
import datasets
import requests
import json
import os

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status() 

        with open(save_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        raise
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
        raise
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")
        raise
    except:
        print(f"Something went wrong when downloading the image from url {url}")
        raise

def download_image_object(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        feature = datasets.Image()
        return feature.encode_example(image)
    else:
        print(f"Something went wrong when downloading the image from url {url}")
        raise

def check_addresses_exist(save_address, dataset_path):
    if not os.path.isdir(save_address):
        print(f"Error: The save_address '{save_address}' does not exist.")
        exit()

    if not os.path.isfile(dataset_path):
        print(f"Error: The dataset file '{dataset_path}' does not exist.")
        exit()

def create_necessary_directories(save_address, dataset_dir_name):
    llava_dataset_path = os.path.join(save_address, dataset_dir_name)
    images_path = os.path.join(llava_dataset_path, 'images')

    if not os.path.exists(llava_dataset_path):
        os.makedirs(llava_dataset_path)
        print(f"Created directory: {llava_dataset_path}")

    if 'llava' in dataset_dir_name:
        if not os.path.exists(images_path):
            os.makedirs(images_path)
            print(f"Created directory: {images_path}")

    return llava_dataset_path, images_path

def read_dataset_file(dataset_path):
    with open(dataset_path, 'r') as file:
        jdata = json.load(file)

    return jdata

def concatenate_captions(img_captions, meme_captions):
    all_captions = ""

    for img_caption in img_captions:
        all_captions += img_caption.strip()
        if not img_caption.endswith(('.', '!', '?')):
            all_captions += '.'   
        all_captions += ' '

    for meme_caption in meme_captions:
        all_captions += meme_caption.strip()
        if not meme_caption.endswith(('.', '!', '?')):
            all_captions += '.'      
        all_captions += ' '

    return all_captions[:-1]