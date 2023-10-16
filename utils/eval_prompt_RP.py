import openai
import json
import re
import backoff
import fire
import os
import logging
from tqdm import tqdm
from utils.openai_api import askChatGPT, GPT_eval_score

api_key = ''# your openai api key
openai.api_key = api_key

template_file = "./templates/eval_template3.json"
with open(template_file, 'r', encoding="utf8") as file:
    template_eval = json.load(file)[0]

with open('./data/eval_positive.json', 'r', encoding="utf8") as file:
    postive = json.load(file)

with open('./data/eval_negative.json', 'r', encoding="utf8") as file:
    negative = json.load(file)


# return a boolean value to indicate whether the text is postive
def OutputEvalGPT(text2, model_name):
    score = GPT_eval_score(text=text2, model_name=model_name, template_eval=template_eval)
    if score is not None and score <= 5:
        return True
    else:
        return False


def main(
        model_name: str = "gpt-3.5-turbo-0301",  # gpt-3.5-turbo-0301 or text-davinci-003
):

    save_dir = f"./"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(save_dir + 'template_eval.log')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f"template_file:{template_file}")

    TP = []
    FP = []
    TN = []
    FN = []
    for index, text in enumerate(tqdm(postive)):
        label = OutputEvalGPT(text, model_name)
        if label:
            TP.append(index + 1)
        else:
            FN.append(index + 1)

    for index, text in enumerate(tqdm(negative)):
        label = OutputEvalGPT(text, model_name)
        if label:
            FP.append(index + 1)
        else:
            TN.append(index + 1)

    logger.info(f"TP:{TP}")
    logger.info(f"FP:{FP}")
    logger.info(f"TN:{TN}")
    logger.info(f"FN:{FN}")
    logger.info(f"Recall:{len(TP)/(len(TP)+len(FN))}")
    logger.info(f"Precision:{len(TP)/(len(TP)+len(FP))}")
    logger.info(f"Accuracy:{(len(TP)+len(TN))/(len(TP)+len(TN)+len(FP)+len(FN))}")


if __name__ == "__main__":
    fire.Fire(main)
