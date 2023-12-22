from utils import (
    check_addresses_exist,
    create_necessary_directories, read_dataset_file,
    read_captions_file
)
from tqdm import tqdm
import argparse
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process meme caption dataset.")
    
    parser.add_argument('--annotation_address', type=str, help='Path to annotation file')
    parser.add_argument('--caption_address', type=str, help='Path to meme captions file')
    parser.add_argument('--output_location', type=str, help='Path to the output directory')

    args = parser.parse_args()

    annotation_address = args.annotation_address or './data/subtask2a/validation.json'
    caption_address = args.caption_address or './captions/llm_finetuned_valid.jsonl'
    output_location = args.output_location or './'
    
    check_addresses_exist(annotation_address, output_location, caption_address=caption_address)
    vicuna_dataset_path, _ = create_necessary_directories(output_location, 'pesuation_vicuna_dataset', False)

    persuation_dataset = read_dataset_file(annotation_address)
    persuasion_captions = read_captions_file(caption_address)
    vicuna_dataset = []

    prompt_template = """
        There is a meme with text written inside the meme: \"<MemeText>\". \
        Also the caption of the meme is: \"<MemeCaption>\". \
        Given a meme caption and the text written in it, predict the logical fallacies and emotional persuasion techniques class labels used in this meme. Here are the classes: \n \
        1-Repetition, 2-Obfuscation, Intentional vagueness, Confusion, 3-Reasoning, 4-Simplification, 5-Causal Oversimplification, \
        6-Black-and-white Fallacy/Dictatorship, 7-Thought-terminating cliché, 8-Distraction, 9-Misrepresentation of Someone's Position (Straw Man), \
        10-Presenting Irrelevant Data (Red Herring), 11-Whataboutism, 12-Justification, 13-Slogans, 14-Bandwagon, 15-Appeal to authority, \
        16-Flag-waving, 17-Appeal to fear/prejudice, 18-Appeal to authority, 19-Glittering generalities (Virtue), 20-Doubt, 21-Name calling/Labeling, \
        22-Smears, 23-Reductio ad hitlerum, 24-Transfer, 25-Exaggeration/Minimisation, 26-Loaded Language, 27-Appeal to (Strong) Emotions. \n \
        Check each one of these technique and if that was used in the meme, put them in output and separate them with <sep>.
    """
    prompt_template = prompt_template.replace("  ", " ").replace("   ", " ").strip()

    i = 0
    question_id = 1
    for d in tqdm(persuation_dataset):   
        image_text = d["text"].replace("\\n", "\n").strip("\n").replace("\n", " \n ").strip()

        vicuna_dataset.append({
            "question_id": question_id,
            "category": "reasoning",
            "turns": [
                prompt_template.replace("<MemeText>", image_text).replace("<MemeCaption>", persuasion_captions[i]["text"])
            ]
        })
        
        i += 1
        question_id += 1

    print("Saving vicuna dataset ...")
    with open(os.path.join(vicuna_dataset_path, 'vicuna_dataset.jsonl'), 'w') as vicuna_dataset_file:
        for obj in vicuna_dataset:
            json_line = json.dumps(obj)
            vicuna_dataset_file.write(json_line + '\n')