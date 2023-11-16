# SAP

This is the official repo of the paper "Attack Prompt Generation for Red Teaming and Defending Large Language Models" accepted to Findings of EMNLP 2023.

## Environment Requirement

The code runs well under python 3.8.0. The required packages are as follows:

- openai == 0.27.4
- backoff == 2.2.1
- fire == 0.5.0
- transformers == 4.28.1
- peft == 0.3.0
- datasets == 2.11.0
- torch == 1.13.1

## Data

We put SAP dataset in `./datasets/`. 

## Run the Code

### Attack Framework

You can run `attack.py` to generate your own SAP dataset with Attack Framework. An example command of SAP5 generation can be found in `case_generate.sh`.

### Defense Framework

An example script for a fine-tuning iteration can be found in `defense_example.sh`.

# Acknowledgment
Some parts of this repository are adopted from alpaca-lora, you can find more information in https://github.com/tloen/alpaca-lora. Thanks for the contributions!
