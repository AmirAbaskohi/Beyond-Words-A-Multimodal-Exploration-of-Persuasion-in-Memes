from utils import (
    check_addresses_exist,
    create_necessary_directories, read_dataset_file,
    create_persuasion_label, copy_image
)
from tqdm import tqdm
import argparse
import uuid
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process meme caption dataset.")
    
    parser.add_argument('--annotation_address', type=str, help='Path to annotation file')
    parser.add_argument('--images_address', type=str, help='Path to images directory')
    parser.add_argument('--output_location', type=str, help='Path to the output folder')

    args = parser.parse_args()

    annotation_address = args.annotation_address or './data/subtask2a/train.json'
    images_address = args.images_address or './train_images'
    output_location = args.output_location or './'
    
    check_addresses_exist(annotation_address, output_location, images_address=images_address)
    llava_dataset_path, llava_data_images_path = create_necessary_directories(output_location, 'pesuation_llava_dataset', True)

    persuation_dataset = read_dataset_file(annotation_address)
    llava_dataset = []

    prompt_template = """
        <image>This is a meme with the following text written inside the meme: \"<MemeText>\". \
        Given a meme and its text, predict the logical fallacies and emotional persuasion techniques class labels used in this meme. Here are the classes: \n \
        1-Repetition, 2-Obfuscation, Intentional vagueness, Confusion, 3-Reasoning, 4-Simplification, 5-Causal Oversimplification, \
        6-Black-and-white Fallacy/Dictatorship, 7-Thought-terminating cliché, 8-Distraction, 9-Misrepresentation of Someone's Position (Straw Man), \
        10-Presenting Irrelevant Data (Red Herring), 11-Whataboutism, 12-Justification, 13-Slogans, 14-Bandwagon, 15-Appeal to authority, \
        16-Flag-waving, 17-Appeal to fear/prejudice, 18-Appeal to authority, 19-Glittering generalities (Virtue), 20-Doubt, 21-Name calling/Labeling, \
        22-Smears, 23-Reductio ad hitlerum, 24-Transfer, 25-Exaggeration/Minimisation, 26-Loaded Language, 27-Appeal to (Strong) Emotions. \n \
        Check each one of these technique and if that was used in the meme, put them in output and separate them with <sep>.
    """
    prompt_template = prompt_template.replace("  ", " ").replace("   ", " ").strip()

    for d in tqdm(persuation_dataset):   
        generated_id = str(uuid.uuid4())
        image_format = d["image"].split(".")[-1]
        image_text = d["text"].replace("\\n", "\n").strip("\n").replace("\n", " \n ").strip()

        copy_image(images_address, llava_data_images_path, d["image"], f"{generated_id}.{image_format}")

        llava_dataset.append({
            "id": generated_id,
            "image": f"{generated_id}.{image_format}",
            "conversations": [
                {
                    "from": "human",
                    "value": prompt_template.replace("<MemeText>", image_text)
                },
                {
                    "from": "gpt",
                    "value": create_persuasion_label(d["labels"])
                }
            ]
        })

    print("Saving llava dataset ...")
    with open(os.path.join(llava_dataset_path, 'llava_dataset.json'), 'w') as llava_dataset_file:
        json.dump(llava_dataset, llava_dataset_file, indent=4)