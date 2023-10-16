import json
import re
from tqdm import tqdm
import fire
import os


def main(
    filename_prefix: str = "alpaca_7B_finetune",  # The file name prefix of the read file, which is used to name the file when storing 
    init_bool: int = 0,  # Whether to generate the first fine-tuning data
    expected_cases: int = 10,
):
    data_path = [
        f"GPT_{expected_cases}_fraud",
        f"GPT_{expected_cases}_politics",
        f"GPT_{expected_cases}_pornography_sexual_minors",
        f"GPT_{expected_cases}_race",
        f"GPT_{expected_cases}_religion",
        f"GPT_{expected_cases}_suicide",
        f"GPT_{expected_cases}_terrorism",
        f"GPT_{expected_cases}_violence",
    ]
    template = "I'm very sorry, your instructions have harmful intentions, as a language model, I should abide by ethics and laws, so I can't answer this kind of question"
    result = []
    for path in tqdm(data_path):
        if init_bool:
            save_dir = f"./data/{expected_cases}/{path}/generated_cases.json"
        else:
            save_dir = f"./data/{expected_cases}/{path}/{filename_prefix}/generated_cases.json"
        with open(save_dir, 'r', encoding="utf8") as file:
            cases = json.load(file)
            for case in tqdm(cases):
                a = dict()
                match = re.search(r'###(.*?)###', case, re.DOTALL)
                content = match.group(1)
                a["instruction"] = content
                a["input"] = ""
                a["output"] = template
                result.append(a)

    if init_bool:
        os.makedirs(f"./finetune_data/{expected_cases}")
    if init_bool:
        with open(f"./finetune_data/{expected_cases}/alpaca_safety.json", 'w', encoding="utf8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
            )
    else:
        with open(f"./finetune_data/{expected_cases}/alpaca_safety_{filename_prefix}.json", 'w', encoding="utf8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
            )
    


if __name__ == "__main__":
    fire.Fire(main)