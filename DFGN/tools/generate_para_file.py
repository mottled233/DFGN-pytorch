import json


if __name__ == "__main__":
    with open("../data/train_name.json", 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    paras_data = {}
    for data in full_data:
        key = data['_id']
        paras = data['context']
        paras_data[key] = paras

    with open("../data/para.json", 'w', encoding='utf-8') as reader:
        data = json.dumps(paras_data, indent=4, ensure_ascii=False)
        reader.write(data)
