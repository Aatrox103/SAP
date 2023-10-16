import os
import fire
import re
import pandas as pd


def main(
        expected_cases: str = "kang",  #
        whether_test_case: int = 0):
    if whether_test_case:
        suffix = ["_5", "_10", "_30"]
    else:
        suffix = [""]
    # ori_col = ['class', '7B', '7B_finetune', '7B_finetune1', '7B_finetune2', '7B_finetune3', '13B', '13B_finetune', '13B_finetune1', '13B_finetune2', '13B_finetune3']
    ori_col = ['class', '7B', '7B_finetune_without_regen', '7B_finetune1_without_regen', '7B_finetune2_without_regen', '7B_finetune3_without_regen','13B', '13B_finetune_without_regen', '13B_finetune1_without_regen', '13B_finetune2_without_regen', '13B_finetune3_without_regen']
    for su in suffix:
        col = [x + su if x != 'class' and x!='7B' and x!='13B' else x for x in ori_col]
        df = pd.DataFrame(columns=col)
        if expected_cases=="kang":
            paths = ["data/from_kang_harmful",'data/from_kang_not_harmful']
        else:
            paths = [f"./data/{expected_cases}/"]
        for path in paths: 
            if expected_cases.isdigit():
                dirs = os.listdir(path)
                for dir in dirs:
                    with open(path + dir + "/avg_score.log", mode="r") as f:
                        data = [dir]
                        result = f.read()
                        for pre in col:
                            if pre == 'class':
                                continue
                            match = re.search(f'{pre}_output_.*:(\d+\.\d\d)', result)
                            score = float(match.group(1))
                            data.append(score)

                        new_row = pd.Series(data, index=df.columns)
                        df = df.append(new_row, ignore_index=True)
            else:
                with open(path + "/avg_score.log", mode="r") as f:
                    data = [path.split("/")[-1]]
                    result = f.read()
                    for pre in col:
                        if pre == 'class':
                            continue
                        match = re.search(f'{pre}_output_.*:(\d+\.\d\d)', result)
                        score = float(match.group(1))
                        data.append(score)

                    new_row = pd.Series(data, index=df.columns)
                    df = df.append(new_row, ignore_index=True)

        df.to_excel(f"./score/{expected_cases}{su}.xlsx", index=False)
        df.to_csv(f"./score/{expected_cases}{su}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)