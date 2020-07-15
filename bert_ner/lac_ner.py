import json as js
from LAC import LAC
from tqdm import tqdm


def clean(string):
    # 用替换空字符串，不然模型会报错
    if string == "":
        return '空字符串'
    else:
        return string

def partition(lst, n):
    # 将数据分割成等份
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


def lac_ner(src_data_path, output_dir):
    with open(src_data_path, 'r', encoding='utf-8') as f:
        datas_qat = js.load(f)

    datas_list = partition(datas_qat, int(len(datas_qat)/1000)+1)

    ENTITY_KIND = ('nw', 'nz', 'PER', 'LOC', 'ORG', 'TIME')
    # ('nw', 'nz', 'PER', 'LOC', 'ORG', 'TIME', 'n')
    query_entities_json = {}
    entities_json = {}
    lac = LAC(mode='lac')

    for datas in tqdm(datas_list):
        querys = []
        contexts = []
        for data in datas:
            key = str(data['_id'])
            context = data['context']
            query = data['question']
            context = "".join(context[0][1])
            querys.append(clean(query))
            contexts.append(clean(context))

        query_results = lac.run(querys)
        context_results = lac.run(contexts)

        for data_index, data in enumerate(datas):
            data_id = str(data['_id'])
            query_result = query_results[data_index]
            context_result = context_results[data_index]
            data_item = {}
            title = data['context'][0][0]
            query_entity_list = []
            entity_list = []
            for q_or_c, result in enumerate([query_result, context_result]):
                kinds = result[1]
                for index, kind in enumerate(kinds):
                    if kind in ENTITY_KIND:
                        entity = result[0][index]
                        if q_or_c == 0:
                            if [entity, kind] not in query_entity_list:
                                query_entity_list.append([entity, kind])
                        else:
                            if [entity, kind] not in entity_list:
                                entity_list.append([entity, kind])
            query_entities_json[data_id] = query_entity_list
            data_item[title] = entity_list
            entities_json[data_id] = data_item

    with open(output_dir +'/'+ "query_entity.json", 'w', encoding='utf-8') as f:
        js.dump(query_entities_json, f)

    with open(output_dir +'/'+ "entity.json", 'w', encoding='utf-8') as f:
        js.dump(entities_json, f)

if __name__ == "__main__":
    lac_ner("data/dev_name.json","output")