import json
import os

def reformat_json_with_id(file_path):

    with open(file_path, 'r',encoding="utf-8") as file:
        data = json.load(file)
    

    formatted_data = []
    for index, entry in enumerate(data):
        parts = entry.split("###")
        attack_prompt = parts[1].strip()
        explanation = parts[2].split("Explanation:", 1)[1].strip()
        
        formatted_entry = {
            "id": index + 1,  
            "Attack Prompt": attack_prompt,
            "Explanation": explanation
        }
        formatted_data.append(formatted_entry)
    

    with open(file_path, 'w') as file:
        json.dump(formatted_data, file, indent=4)

def process_all_json_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'generated_cases.json':
                full_path = os.path.join(root, file)
                print(f"{full_path}")
                reformat_json_with_id(full_path)


process_all_json_files('datasets')
