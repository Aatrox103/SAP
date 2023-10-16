import openai
import backoff
import re


api_key = ''# your openai api key
openai.api_key = api_key

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout))
def askChatGPT(prompt, model_name):
    if model_name == "gpt-3.5-turbo-0301":
        messages = [{"role": "user", "content": prompt}]
        # print(messages[0]['content'])
        response = openai.ChatCompletion.create(model=model_name, messages=messages, temperature=1)
        return response['choices'][0]['message']['content']
    elif model_name == "text-davinci-003":
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=2048,
        )
        message = completions.choices[0].text
        return message


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout))
def Moderation(input):
    response = openai.Moderation.create(model="text-moderation-stable", input=input)
    return response

def GPT_eval_score(text, model_name, template_eval):
    content = askChatGPT(template_eval.format(text), model_name)
    match = re.search(r'(\d+)', content)
    if match is None:
        return None
    else:
        score = match.group(1)
        return int(score)
