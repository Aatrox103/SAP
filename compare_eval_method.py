from googleapiclient import discovery
import googleapiclient
import json
from tqdm import tqdm
from time import sleep
from utils.openai_api import askChatGPT, GPT_eval_score, Moderation
import numpy as np
from sklearn.metrics import roc_auc_score


def analyze_text(text):
    API_KEY = ''# your perspective api key

    att = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT', 'SEXUALLY_EXPLICIT', 'FLIRTATION']
    client = googleapiclient.discovery.build(
        'commentanalyzer',
        'v1alpha1',
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {'comment': {'text': text}, 'requestedAttributes': {x: {} for x in att}}

    response = client.comments().analyze(body=analyze_request).execute()

    toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
    severe_toxicity_score = response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']

    max_score = 0
    for x in att:
        if response['attributeScores'][x]['summaryScore']['value'] > max_score:
            max_score = response['attributeScores'][x]['summaryScore']['value']
    return max_score

def generate_perspective_api_score():
    outscore = []
    with open('data/eval_negative.json', mode='r', encoding="utf8") as f:
        outputs = json.load(f)
        for output in tqdm(outputs):
            sleep(1)
            outscore.append(analyze_text(output))

    with open('data/p_api_negative_score.json', mode='w', encoding="utf8") as f:
        json.dump(
            outscore,
            f,
            ensure_ascii=False,
        )

    outscore = []
    with open('data/eval_positive.json', mode='r', encoding="utf8") as f:
        outputs = json.load(f)

        for output in tqdm(outputs):
            sleep(1)
            outscore.append(analyze_text(output))

    with open('data/p_api_positive_score.json', mode='w', encoding="utf8") as f:
        json.dump(
            outscore,
            f,
            ensure_ascii=False,
        )

def generate_GPT_score():
    outscore = []
    with open('./templates/eval_template2.json', 'r', encoding="utf8") as file:
        template_eval = json.load(file)[0]
    with open('data/eval_negative.json', mode='r', encoding="utf8") as f:
        outputs = json.load(f)
        for output in tqdm(outputs):
            fail_times=0
            while True:
                score = GPT_eval_score(text=output, template_eval=template_eval, model_name="gpt-3.5-turbo-0301")
                if score is not None:
                    outscore.append(score)
                    break
                else:
                    fail_times += 1
                    if fail_times >= 5:
                        outscore.append(10)
                        break

    with open('data/gpt_negative_score.json', mode='w', encoding="utf8") as f:
        json.dump(
            outscore,
            f,
            ensure_ascii=False,
        )

    outscore = []
    with open('data/eval_positive.json', mode='r', encoding="utf8") as f:
        outputs = json.load(f)
        for output in tqdm(outputs):
            fail_times=0
            while True:
                score = GPT_eval_score(text=output, template_eval=template_eval, model_name="gpt-3.5-turbo-0301")
                if score is not None:
                    outscore.append(score)
                    break
                else:
                    fail_times += 1
                    if fail_times >= 5:
                        outscore.append(10)
                        break


    with open('data/gpt_positive_score.json', mode='w', encoding="utf8") as f:
        json.dump(
            outscore,
            f,
            ensure_ascii=False,
        )


if __name__ == '__main__':
    # generate_perspective_api_score()
    # generate_GPT_score()

    with open('data/p_api_negative_score.json', mode='r', encoding="utf8") as f:
        negative_samples = json.load(f)

    with open('data/p_api_positive_score.json', mode='r', encoding="utf8") as f:
        positive_samples = json.load(f)

    all_samples = -np.array(positive_samples + negative_samples)

    # Create labels: 1 for positive samples, 0 for negative samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Calculate AUC using roc_auc_score
    auc = roc_auc_score(labels, all_samples)
    print("perspective_api_AUC:", auc)


    with open('data/gpt_negative_score.json', mode='r', encoding="utf8") as f:
        negative_samples = json.load(f)

    with open('data/gpt_positive_score.json', mode='r', encoding="utf8") as f:
        positive_samples = json.load(f)

    all_samples = -np.array(positive_samples + negative_samples)

    # Create labels: 1 for positive samples, 0 for negative samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Calculate AUC using roc_auc_score
    auc = roc_auc_score(labels, all_samples)
    print("GPT_AUC:", auc)

#       perspective_api_AUC: 0.855
#       GPT_AUC: 0.9521999999999999