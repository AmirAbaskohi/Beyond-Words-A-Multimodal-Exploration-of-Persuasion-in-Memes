# <p align="center">Beyond Words: A Multimodal and Multilingual Exploration of Persuasion in Memes</p>

<h2 align="center">
  <p><a href="https://semeval.github.io/SemEval2024">[SemEval@NAACL 2024]</a> Beyond Words: A Multimodal and Multilingual Exploration of Persuasion in Memes</p>
</h2>

<p align="center">
  <br>
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href="#"><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000"></a>
  <a href="https://huggingface.co/AmirHossein1378/LLaVA-1.5-7b-meme-captioner"><img alt="Video" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue"></a>
  <a href="#"><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=googlesheets&logoColor=FFBB00"></a>
</p>

## Intro
This repo covers the implementation of the following paper:  **[BCAmirs at SemEval-2024 Task 4: Beyond Words: A Multimodal and Multilingual Exploration of Persuasion in Memes]()** by [Amirhossein Abaskohi](https://amirabaskohi.github.io/), [Amirhossein Dabiriaghdam](https://dabiriaghdam.ir/), [Lele Wang](https://ece.ubc.ca/lele-wang/), and [Giuseppe Carenini](https://www.cs.ubc.ca/~carenini/), accepted to SemEval@NAACL 2024.

## Abstract
Memes, combining text and images, frequently use metaphors to convey persuasive messages, shaping public opinion. Motivated by this, our team engaged in SemEval-2024 Task 4, a hierarchical multi-label classification task designed to identify rhetorical and psychological persuasion techniques embedded within memes. To tackle this problem, we introduced a caption generation step to assess the modality gap and the impact of additional semantic information from images, which improved our result. Our best model utilizes GPT-4 generated captions alongside meme text to fine-tune RoBERTa as the text encoder and CLIP as the image encoder. It outperforms the baseline by a large margin in all 12 subtasks. In particular, it ranked in top-3 across all languages in Subtask 2a, and top-4 in Subtask 2b, demonstrating quantitatively strong performance. The improvement achieved by the introduced intermediate step is likely attributable to the metaphorical essence of images that challenges visual encoders. This highlights the potential for improving abstract visual semantics encoding.

## Competition Results

Results of our best model (at the time of submitting evaluation results) on the test dataset of different subtasks. In the table, the ranking and the values are based on hierarchical F1. It can be seen that our method performed well in the Subtask 2 (our main focus) for all four languages, and also the English subset of the first subtask. Our model struggled with non-English subsets of Subtask 1 since (I) we did not have access to the image of the meme and therefore no caption was available, and (II) our models only understood English, so we relied on a translation (using Google Translate) of the memes' text.

| Subtask                 | Ours     | Baseline | Rank |
|-------------------------|----------|----------|------|
| 2a - English            | **70.497** | 44.706   | 3    |
| 2a - Bulgarian          | **62.693** | 50.000   | 1    |
| 2a - North Macedonian   | **63.681** | 55.525   | 1    |
| 2a - Arabic             | **52.613** | 48.649   | 1    |
| 2b - English            | **80.337** | 25.000   | 4    |
| 2b - Bulgarian          | **64.719** | 16.667   | 4    |
| 2b - North Macedonian   | **64.719** | 09.091   | 4    |
| 2b - Arabic             | **61.487** | 22.705   | 1    |
| 1 - English             | **69.857** | 36.865   | 2    |
| 1 - Bulgarian           | **44.834** | 28.377   | 13   |
| 1 - North Macedonian   | **39.298** | 30.692   | 12   |
| 1 - Arabic             | **39.625** | 35.897   | 9    |

## Method
Building upon prior research on MLLMs, the prevailing method involves tokenizing image concepts and conveying these tokens alongside textual tokens to a language model. While these models possess the ability to impart more semantic information from the image, their focus typically centers on identifying objects and their relationships within the image. Consequently, this study explores the impact of initially prompting the model to generate descriptive information aimed at conveying semantic context. We utilize this information for data classification, comparing it to the conventional approach of fine-tuning an end-to-end model.

In this paper we used three different models for generating meme captions: BLIP-2, LLaVA-1.5-7B, and GPT-4. We fine-tuned BLIP-2 and LLaVA-1.5 for generating captions and used GPT-4 in zero-shot settings. LLaVA-1.5 outperforms BLIP-2 in the quality of generated captions. In order to fine-tune our meme captioning model, we used MemeCap. The following figure, depicts the supervised fine-tuning loop of the LLaVA-1.5-7B model on the MemeCap dataset for caption generation. The OCR module extracts text from the meme images. The vision encoder (CLIP), a frozen component of LLaVA-1.5-7B, processes the meme images. The vision-language projector bridges the gap between CLIP’s representation and the embedding space of Vicuna. While CLIP remains frozen, the vision-language projector is finetuned. The Vicuna component was experimented with both frozen and finetuned setups to generate captions for the memes.

![image](https://github.com/AmirAbaskohi/Beyond-Words-A-Multimodal-Exploration-of-Persuasion-in-Memes/assets/50926437/b848eddf-4aa0-4bd3-85e5-2a051c9e97a8)

The following figure illustrates the architecture of ConcatRoBERTa, our best-performing model. The GPT4-V(ision) component generates a descriptive caption of the meme image. The caption is then combined with the text written in
the meme, which is processed by the RoBERTa. The Vision encoder utilizes a pretrained vision transformer model (CLIP-ViT), to encode and analyze the visual elements of the meme. The MLP Classifier takes the combined visual and textual representations and classifies the meme. The key innovation of ConcatRoBERTa is its caption generation intermediate step which extracts more informative data from the meme image and the ability to effectively integrate and leverage both visual and textual modalities in a unified architecture, enabling comprehensive understanding and analysis of memes, which often rely on the complex interplay between images and text. The RoBERTa and MLP classifiers are finetuned, While CLIP remains frozen.

![image](https://github.com/AmirAbaskohi/Beyond-Words-A-Multimodal-Exploration-of-Persuasion-in-Memes/assets/50926437/8e3a2d1b-7ac5-48c6-b15d-8f66f091f279)


## Dataset

The main dataset of the task is not publicly available and you need to send a request to the task organizers of the task. You can find task organizers using [this link](https://propaganda.math.unipd.it/semeval2024task4/). Also we utilzed the [MemeCap](https://aclanthology.org/2023.emnlp-main.89/) dataset as a part of our experiments.

To fine-tune/use the models, you need to transform the datasets into their required format. Use the scripts in the "DatasetCreator" directory to generate them. To understand the needed parameters, run the following command for each script:

```
python3 script.py -h
```

For some parts, you may require our generated captions. In case needed, you can contact us (the authors) for the generated captions.

## How to run?

### Meme Captioning Model

Our meme captioner model, i.e., fine-tuned LLaVA-1.5-7B, is released [here on huggingface](https://huggingface.co/AmirHossein1378/LLaVA-1.5-7b-meme-captioner). To run it follow these steps:

1. Clone this repository and navigate to LLaVA folder
```
git clone https://github.com/AmirAbaskohi/Beyond-Words-A-Multimodal-Exploration-of-Persuasion-in-Memes.git
cd LLaVA
```
2. Run the following commands:
```
conda create -n llava_captioner python=3.8 -y
conda activate llava_captioner
pip3 install -e .
pip3 install transformers==4.31.0 
pip3 install protobuf
````
3. Finally you can chat with the model through CLI by passing our model as the model path:
```
python3 -m llava.serve.cli  --model-path AmirHossein1378/LLaVA-1.5-7b-meme-captioner    --image-file PATH_TO_IMAGE_FILE
```

For more information about different ways to run LLaVA please refer to original LLaVA repository [here](https://github.com/haotian-liu/LLaVA).

### Training and Inference

After generating the required dataset with suitable data format for each model using the "DatasetCreator" scripts, use the following commands for each model:

#### ConcatRoBERTa

##### Fine-tune
First change the following lines based on your inputs (you can change the hyper-parameters in the code as well):
```
Line 37: args.data_path = IMAGES_PATH
.
.
.
Line 84: raw_train_ds = TRAIN_CSV_FILE
Line 85: raw_val_ds = VALIDATION_CSV_FILE
```

Also if you want to decide whether to use the generated "captions" during training phase, change `no_caption` argument in `preprocess_function`.

Then run the following command:

```
python3 train.py
```

##### Evaluation
First change the following lines based on your inputs:
```
Line 39: args.data_path = IMAGES_PATH
.
.
.
Line 86: raw_val_ds = VALIDATION_CSV_FILE
.
.
.
Line 295: VALIDATION_JSON_FILE
Line 301: OUTPUT_JSON_FILE
```

Then run the following command:

```
python3 eval.py
```

#### LLaVA

##### Fine-tune
```
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path PATH_TO_DATA \
    --image_folder PATH_TO_IMAGES \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-SemEval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --tune_mm_mlp_adapter False
```

##### Evaluation
```
python3 LLaVA/llava/eval/model_vqa.py \
    --model-path LLaVA/checkpoints/llava-v1.5-7b-SemEval \
    --model-base liuhaotian/llava-v1.5-7b \
    --image-folder PATH_TO_IMAGES \
    --question-file PATH_TO_QUESTION_JSONL \
    --answers-file PATH_TO_ANSWER_JSONL
```

#### RoBERTa
##### Fine-tune
First change the following lines based on your inputs (you can change the hyper-parameters in the code as well):
```
LINE 52: TRAIN_JSONL_FILE
LINE 53: VALIDATION_JSONL_FILE
.
.
.
LINE 95: MODEL_CHECKPOINT_SAVE_PATH
```

Also if you want to decide whether to use the generated "captions" during training phase, change `no_caption` argument in `preprocess_function`.

Then run the following command:

```
python3 train.py
```

##### Evaluation

First change the following lines based on your inputs:

```
LINE 55: VALIDATION_JSONL_FILE
.
.
.
LINE 76: MODEL_CHECKPOINT_PATH
.
.
.
LINE 123: OUTPUT_FILE_PATH
```

Then run the following command:

```
python3 eval.py
```

#### Vicuna

##### Fine-tune

```
deepspeed fastchat/train/train_lora.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5  \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path PATH_TO_DATA \
    --output_dir ./checkpoints/vicuna-7b-v1.5-SemEval \
    --num_train_epochs 1 \
    --fp16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 51 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed FastChat/playground/deepspeed_config_s3.json \
    --gradient_checkpointing True \
    --flash_attn False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
```

##### Evaluation

For the input file, you should first change the following line:

```
Line 281: question_file = PATH_TO_QUESTION_JSONL
```

Then you can run the following command:

```
python3 gen_model_answer.py \
    --model-path FastChat/checkpoints/vicuna-7b-v1.5-SemEval \
    --model-id vicuna-7b-v1.5-SemEval \
    --max-new-token 100
```

#### VisualBERT

You should first run the `feature_extractor.py`. To do so first change the following lines based on your inputs:
```
LINE 204: IMAGE_PATH
.
.
.
LINE 247: EXTRACTED_FEATURES_PATH
```

Then run the following command:

```
python3 feature_extractor.py
```

##### Fine-tune
First change the following lines based on your inputs (you can change the hyper-parameters in the code as well):
```
LINE 60: EXTRACTED_FEATURES_VALIDATION
LINE 63: EXTRACTED_FEATURES_TRAIN
.
.
.
LINE 111: TRAIN_CSV_FILE
LINE 112: VALIDATION_CSV_FILE
```

Also if you want to decide whether to use the generated "captions" during training phase, change `no_caption` argument in `preprocess_function`.

Then run the following command:

```
python3 main.py
```

##### Evaluation
First change the following lines based on your inputs:
```
LINE 50: MODEL_CHECKPOINT_PATH
LINE 55: EXTRACTED_FEATURES_VALIDATION
LINE 58: EXTRACTED_FEATURES_TRAIN
.
.
.
LINE 106: VALIDATION_CSV_FILE
.
.
.
LINE 175: VALIDATION_JSON_FILE
LINE 181: OUTPUT_FILE_PATH
```

Then run the following command:

```
python3 eval.py
```

#### BLIP-2
##### Fine-tune
First change the following lines based on your inputs (you can change the hyper-parameters in the code as well):
```
LINE 12: TRAIN_DATASET_PATH
.
.
.
LINE 15: VALIDATION_DATASET_PATH
```
Then run the following command:

```
python3 train.py
```

##### Evaluation

First change the following lines based on your inputs:

```
LINE 13: VALIDATION_DATASET_PATH
.
.
.
LINE 49: MODEL_PATH
```

Then run the following command:

```
python3 eval.py
```

Also, to evaluate the generated captions you can run the following python script in the MemeCap folder:
```
python3 calc_metric.py
```
But first you may need to update the following lines:
```
LINE 7: MODEL_OUTPUT_JSONL
.
.
.
LINE 10: GROUND_TRUTH_JSON
```

## Results

### Meme Captioning

The following table is the performance comparison of meme captioning models on MemeCap test set. In this table `+ OCR data` means for the training data we also appended the extracted text from the meme to help with the task of captioning the memes. The fine-tuned versions of the models yield superior captions, with all LLaVA iterations outperforming BLIP. The most effective model is LLaVA when both the language model and projector are tuned, particularly when incorporating text within the image generated by the OCR model. You can find this model [here](https://huggingface.co/AmirHossein1378/LLaVA-1.5-7b-meme-captioner).

| Model                                           | F1-Bertscore | ROUGE-L | BLEU-4 |
|-------------------------------------------------|--------------|---------|--------|
| BLIP-2 (fine-tuned)                            | 58.00        | 26.39   | 47.93  |
| LLaVA-1.5 (projector fine-tuned)               | 59.01        | 27.41   | **57.78** |
| LLaVA-1.5 (LLM & projector fine-tuned)         | 59.23        | 27.40   | 45.53  |
| LLaVA-1.5 (projector fine-tuned + OCR data)    | **59.80**    | **28.08**   | 53.33  |
| LLaVA-1.5 (LLM & projector fine-tuned + OCR data) | **59.90**    | 27.86   | 53.86  |
| BLIP-2 (zero-shot)                             | 50.30        | 12.88   | 31.81  |
| LLaVA-1.5 (zero-shot)                          | 55.11        | 19.31   | 40.15  |


### Persuasion Technique Classification

The following table is the comparison of results of different methods on dev set of Subtask 2a. In table columns, H-F1, H-Precision,
and H-Recall, are hierarchical-F1, -precision, and -recall respectively. As can be seen, expectedly, models prefer to
receive more information about the image, and models incorporating all features (e.g., text, caption, and image) tend
to perform better. However, captions appear to be more informative. This suggests that although some information
from the image may not be fully conveyed through text, utilizing models to initially analyze the image, particularly in
meme tasks like this, and then prompting them to make decisions based on that analysis, yields superior performance
compared to making decisions without leveraging their full capabilities.

| Model                                           | H-F1    | H-Precision | H-Recall |
|-------------------------------------------------|---------|-------------|----------|
| LLaVA-1.5 (image)                              | 58.21   | 62.74       | 54.31    |
| LLaVA-1.5 (image+text)                         | 62.59   | 66.00       | 59.51    |
| LLaVA-1.5 (image+text+caption from LLaVA-1.5)  | 63.33   | 67.02       | 60.02    |
| Vicuna-1.5 (text)                              | 62.69   | 71.03       | 56.10    |
| Vicuna-1.5 (text+caption from LLaVA-1.5)       | 63.11   | 70.86       | 56.88    |
| Vicuna-1.5 (text+caption from GPT-4)           | 65.337  | 75.204      | 57.759   |
| BERT (text)                                    | 64.881  | 75.400      | 56.938   |
| BERT (text+caption from LLaVA-1.5)             | 66.455  | 74.229      | 60.155   |
| BERT (text+caption from GPT-4)                 | 66.829  | 75.958      | 59.659   |
| RoBERTa (text)                                 | 66.740  | 76.846      | 58.983   |
| RoBERTa (text+caption from LLaVA-1.5)          | 67.750  | 73.699      | 62.690   |
| RoBERTa (text+caption from GPT-4)              | 69.913  | 76.999      | 64.021   |
| VisualBERT (image+text)                        | 51.496  | 39.779      | 72.998   |
| VisualBERT (image+text+caption from LLaVA-1.5) | 57.714  | 57.841      | 62.690   |
| ConcatRoBERTa (image+text)                     | 65.188  | 73.443      | 58.601   |
| ConcatRoBERTa (image+text+caption from LLaVA-1.5)| 67.166 | 75.283      | 60.629   |
| ConcatRoBERTa (image+text+caption from GPT-4)   | **71.115** | 76.101      | 66.742   |
| Baseline                                       | 44.706  | 68.778      | 33.116   |

## Acknowledgement

Feel free to consult related work that provides the groundwork for our framework and code repository:

- [Vicuna](https://github.com/lm-sys/FastChat)
- [LLaVA](https://github.com/haotian-liu/LLaVA)


## Citation
If you find our work or models useful for your research and applications, please cite using this BibTeX:

```
@misc{abaskohi2024bcamirs,
      title={BCAmirs at SemEval-2024 Task 4: Beyond Words: A Multimodal and Multilingual Exploration of Persuasion in Memes}, 
      author={Amirhossein Abaskohi and Amirhossein Dabiriaghdam and Lele Wang and Giuseppe Carenini},
      year={2024},
      eprint={2404.03022},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
