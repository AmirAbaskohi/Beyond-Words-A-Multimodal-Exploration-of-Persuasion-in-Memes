import datasets
import json
import os
import shutil
import random

def check_addresses_exist(annotation_address, output_location, images_address=None, caption_address=None):
    if not os.path.isfile(annotation_address):
        print(f"Error: The annotation file '{annotation_address}' does not exist.")
        exit()

    if caption_address != None and (not os.path.isfile(caption_address)):
        print(f"Error: The annotation file '{caption_address}' does not exist.")
        exit()

    if not os.path.isdir(output_location):
        print(f"Error: The output directory '{output_location}' does not exist.")
        exit()

    if images_address != None and (not os.path.isdir(output_location)):
        print(f"Error: The images directory '{images_address}' does not exist.")
        exit()

def create_necessary_directories(save_address, dataset_dir_name, create_images_dir=False):
    dataset_path = os.path.join(save_address, dataset_dir_name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Created directory: {dataset_path}")

    if not create_images_dir:
        return dataset_path, None

    images_path = os.path.join(dataset_path, 'images')
    
    if not os.path.exists(images_path):
        os.makedirs(images_path)
        print(f"Created directory: {images_path}")

    return dataset_path, images_path

def read_dataset_file(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        jdata = json.load(file)

    return jdata

def create_persuasion_label(labels):
    order_table = {
        "Repetition": 1,
        "Obfuscation, Intentional vagueness, Confusion": 2,
        "Reasoning": 3,
        "Simplification": 4,
        "Causal Oversimplification": 5,
        "Black-and-white Fallacy/Dictatorship": 6,
        "Thought-terminating cliché": 7,
        "Distraction": 8,
        "Misrepresentation of Someone's Position (Straw Man)": 9,
        "Presenting Irrelevant Data (Red Herring)": 10,
        "Whataboutism": 11,
        "Justification": 12,
        "Slogans": 13,
        "Bandwagon": 14,
        "Appeal to authority": 15,
        "Flag-waving": 16,
        "Appeal to fear/prejudice": 17,
        "Appeal to authority": 18,
        "Glittering generalities (Virtue)": 19,
        "Doubt": 20,
        "Name calling/Labeling": 21,
        "Smears": 22,
        "Reductio ad hitlerum": 23,
        "Transfer": 24,
        "Exaggeration/Minimisation": 25,
        "Loaded Language": 26,
        "Appeal to (Strong) Emotions": 27
    }

    sorted_labels = sorted(labels, key=lambda x: order_table[x])
    return ("<sep>".join(sorted_labels))

def copy_image(images_directory, output_images_directoy, image_name, new_image_name):
    source_path = os.path.join(images_directory, image_name)
    destination_path = os.path.join(output_images_directoy, new_image_name)

    shutil.copy(source_path, destination_path)

def read_captions_file(file_path):
    json_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            json_list.append(data)
    return json_list

def generate_random_numbers(max_num, n, exclude_list):
    random_numbers = set()
    while len(random_numbers) < n:
        num = random.randint(0, max_num - 1)
        if num not in exclude_list:
            random_numbers.add(num)

    return list(random_numbers)

def create_few_shot_examples(indexes, annotations, captions):
    examples = []
    counter = 1
    for i in indexes:
        example = f"{counter}. \n "
        annotation = annotations[i]["text"].replace("\\n", "\n").strip("\n").replace("\n", " \n ").strip()
        caption = captions[i]["text"]
        example += f"Text Writtent in the meme: {annotation} \n "
        example += f"Caption: {caption} \n "
        example += f"Label: {create_persuasion_label(annotations[i]['labels'])}"
        examples.append(example)
        counter += 1
    return examples