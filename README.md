# MIG-LLM

## Model and Tokenizer Setup

Before fine-tuning, download the initial weights and configuration files of your chosen base model from Hugging Face and save them to the directory specified by `model_name_or_path` in `config/config-meter.json`. Alternatively, update this setting to match your local model directory. Ensure the path is correct before starting fine-tuning.

You can use the model’s official tokenizer or train a custom tokenizer for your task. Set `tokenizer_name` to the tokenizer’s directory and ensure that it is compatible with the model.

## Dataset

The raw datasets are available on Hugging Face: [Maritime-Dataset](https://huggingface.co/datasets/JunhaoJiang/Maritime-Dataset).

The repository contains four CSV files: `ZSI.csv`, `YZR.csv`, `GBW.csv`, and `CBW.csv`. These raw datasets are provided for data analysis, further processing, and subsequent model training. Download and preprocess the data as needed for your task before training.

## Single-Scene Data Example

The [`data_example/`](data_example/) directory provides a data example for a single scene. It includes trajectory data split into training, validation, and test sets, together with associated images and captions, to illustrate the data organization used by this project.

## Workflow

Once all files are in the correct locations and the relevant paths are configured, follow the steps below.

1. **Dataset Construction**

   Run [dataprocess.py](dataprocess.py) to construct the training, validation, and test datasets (Train/Val/Test).

2. **Tokenizer Selection**

   To use the model’s official tokenizer, place its files in the directory specified in the configuration. To create a custom tokenizer, run [tokenizer.py](tokenizer.py) and set `tokenizer_name` to the directory containing the generated tokenizer.

3. **Model Training**

   Run [Training.py](Training.py) to start fine-tuning the model.

4. **Model Testing**

   Run [Testing.py](Testing.py) to evaluate the fine-tuned model.
