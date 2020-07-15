import json

with open("E:/CAIL2020/qat_entity_n.json", 'r', encoding='utf-8') as reader:
    full_data = json.load(reader)

for key in full_data:
    paras = full_data[key]
    for pk in paras:
        new_entities = []
        para = paras[pk]
        for entity in para:
            if entity not in new_entities:
                new_entities.append(entity)
        paras[pk] = new_entities

with open("E:/CAIL2020/qat_entity_form_n.json", 'w', encoding='utf-8') as reader:
    data = json.dumps(full_data, indent=4, ensure_ascii=False)
    reader.write(data)
