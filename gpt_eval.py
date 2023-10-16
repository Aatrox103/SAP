import openai
import json
import re
import os
import backoff
import logging
import fire

from tqdm import tqdm
from utils.openai_api import askChatGPT, GPT_eval_score, Moderation


#data_path: The name of the grandparent directory where the file is read and stored
#gpt_output: list of text to be evaluated
#eval_filename_prefix: The file name prefix of the read file, which is used to name the file when storing the result
#Use gpt to evaluate the harmfulness of the generated results,
#and two files will be generated in different directories of the evaluated file to save the results
def GPT_eval(data_path, gpt_output, model_name, eval_filename_prefix):
    with open('./templates/eval_template2.json', 'r', encoding="utf8") as file:
        template_eval = json.load(file)[0]
    gpt_eval_score = []
    save_dir = f"./data/{data_path}/"
    if eval_filename_prefix is None and os.path.exists(f'{save_dir}gpt_output_score.json'):
        with open(f'{save_dir}gpt_output_score.json', 'r', encoding="utf8") as file:
            gpt_eval_score = json.load(file)
    elif os.path.exists(f'{save_dir}{eval_filename_prefix}/{eval_filename_prefix}_output_score.json'):
        with open(f'{save_dir}{eval_filename_prefix}/{eval_filename_prefix}_output_score.json', 'r', encoding="utf8") as file:
            gpt_eval_score = json.load(file)
    
    if len(gpt_eval_score) == len(gpt_output):
        write_log=False
    else:
        write_log=True
    gpt_output= gpt_output[len(gpt_eval_score):]
    fail_index = []
    for index, output in enumerate(tqdm(gpt_output)):
        fail_times = 0
        while True:
            score = GPT_eval_score(text=output, template_eval=template_eval, model_name=model_name)
            if score is not None:
                gpt_eval_score.append(score)
                break
            else:
                fail_times += 1
                if fail_times >= 5:
                    gpt_eval_score.append(10)
                    fail_index.append(index)
                    break
        if index % 5 == 0:
            save_dir = f"./data/{data_path}/"
            if eval_filename_prefix is None:
                with open(f'{save_dir}gpt_output_score.json', 'w', encoding="utf8") as file:
                    json.dump(
                        gpt_eval_score,
                        file,
                        ensure_ascii=False,
                    )
            else:
                with open(f'{save_dir}{eval_filename_prefix}/{eval_filename_prefix}_output_score.json', 'w', encoding="utf8") as file:
                    json.dump(
                        gpt_eval_score,
                        file,
                        ensure_ascii=False,
                    )

    save_dir = f"./data/{data_path}/"
    if eval_filename_prefix is None:
        with open(f'{save_dir}gpt_output_score.json', 'w', encoding="utf8") as file:
            json.dump(
                gpt_eval_score,
                file,
                ensure_ascii=False,
            )
    else:
        with open(f'{save_dir}{eval_filename_prefix}/{eval_filename_prefix}_output_score.json', 'w', encoding="utf8") as file:
            json.dump(
                gpt_eval_score,
                file,
                ensure_ascii=False,
            )

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(save_dir + 'avg_score.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    if write_log:
        if eval_filename_prefix is None:
            logger.info("gpt_output_avg_score:{:.2f}".format(sum(gpt_eval_score) / len(gpt_eval_score)))
            # logger.info("gpt_output_fail:{}".format(fail_index))
        else:
            logger.info("{}_output_avg_score:{:.2f}".format(eval_filename_prefix, sum(gpt_eval_score) / len(gpt_eval_score)))
            # logger.info("{}_output_fail:{}".format(eval_filename_prefix,fail_index))


#The location of the evaluated json file is f"./data/{data_path}/{eval_filename_prefix}/{eval_filename_prefix}_output.json"
def main(
    data_path="10/GPT_10_fraud",  #The name of the grandparent directory where the file is read and stored
    expected_cases: int = 30,
    eval_method: str = "GPT",  # "GPT" or "openai api"
    eval_filename_prefix: str = None,  # The file name prefix of the read file, which is used to name the file when storing the result
    model_name: str = "gpt-3.5-turbo-0301",
):
    if eval_filename_prefix is None:
        with open(f"./data/{data_path}/gpt_output.json", mode='r', encoding="utf8") as f:
            gpt_output = json.load(f)
    else:
        with open(f"./data/{data_path}/{eval_filename_prefix}/{eval_filename_prefix}_output.json", mode='r', encoding="utf8") as f:
            gpt_output = json.load(f)
    if eval_method == "GPT":
        GPT_eval(data_path=data_path, gpt_output=gpt_output, model_name=model_name, eval_filename_prefix=eval_filename_prefix)


if __name__ == "__main__":
    fire.Fire(main)
