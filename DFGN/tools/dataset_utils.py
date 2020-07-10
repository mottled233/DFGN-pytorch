from pyhanlp import *
import random
from utils import *
from tqdm import tqdm
import numpy as np
import gzip
import pickle


name_list = ["张吉惟", "林国瑞", "林玟书", "江奕云", "刘柏宏", "阮建安", "王帆", "夏志豪", "李中冰",
             "黄文隆", "谢彦文", "傅智翔", "洪振霞", "吕盈", "方强", "黎贵", "郑雯", "雷宝",
             "吴美", "王美珠", "郭芳", "李雅惠", "陈文婷", "曹敏", "王婷", "陈婉", "吴玉", "蔡依婷",
             "林家纶", "黄丽", "黄芸", "吴韵如", "李芬", "李白", "方玉", "刘惠",
             "吴佳", "周白", "张伦", "倪怡芳", "杨芳", "黄文旺",
             "黄盛", "郑丽", "许云", "张孟涵", "王龙", "朱政廷", "邓诗涵", "陈政倩", "吴俊伯",
             "吴思翰", "林佩玲", "邓海", "陈依", "李建", "武淑芬", "金雅琪", "张仪",
             "王俊民", "张诗刚", "林慧颖", "沈俊君", "陈淑妤", "黄彦宜", "周孟儒", "潘欣"]

patterns = [re.compile(r"[\u4e00-\u9fa5][xX×]+\d{1,2}"),
            re.compile(r"[\u4e00-\u9fa5]某[\u4e00-\u9fa5]?")]


def get_random_name(name_to_ori={}):
    """
    Select a random name not in name dict.
    :param name_to_ori: For distinct
    :return: name string
    """
    while True:
        idx = random.randint(0, len(name_list)-1)
        name = name_list[idx]
        if name not in name_to_ori:
            return name


def trans_name(example):
    """
    Convert the desensitized name in an example into a reasonable name
    :param example: An example from origin hotpotQA dataset
    :return: the example converted
    """
    ori_to_name = dict()
    name_to_ori = dict()
    example['context'][0][0] = trans_name_str(example['context'][0][0], ori_to_name, name_to_ori)

    new_context_list = []
    for sent in example['context'][0][1]:
        new_context_list.append(trans_name_str(sent, ori_to_name, name_to_ori))
    example['context'][0][1] = new_context_list

    example['question'] = trans_name_str(example['question'], ori_to_name, name_to_ori)
    example['answer'] = trans_name_str(example['answer'], ori_to_name, name_to_ori)

    for sp in example['supporting_facts']:
        sp[0] = trans_name_str(sp[0], ori_to_name, name_to_ori)

    example['ori_to_name'] = ori_to_name
    example['name_to_ori'] = name_to_ori
    return example


def trans_name_str(sent, ori_to_name={}, name_to_ori={}):
    """
    Convert the desensitized name in one sentence into a reasonable name
    :param sent: Sentence
    :param ori_to_name: Avoid convert same name to different one in context
    :param name_to_ori: Reverse of ori_to_name.
    :return: Converted sentence
    """
    array = []
    for pattern in patterns:
        array.extend(pattern.findall(sent))
    for ori_name in array:
        if ori_name in ori_to_name:
            name = ori_to_name[ori_name]
        else:
            name = get_random_name(name_to_ori)
        sent = sent.replace(ori_name, name)
        ori_to_name[ori_name] = name
        name_to_ori[name] = ori_name
    return sent


def trans_name_file(in_file_name, out_file_name):
    """
    Convert the desensitized name in the CAIL2020 dataset file into a reasonable name
    :param in_file_name:
    :param out_file_name: Output file path
    """
    # clean the information desensitization
    with open(in_file_name, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    for data in tqdm(full_data):
        trans_name(data)

    with open(out_file_name, 'w', encoding='utf-8') as reader:
        data = json.dumps(full_data, indent=4, ensure_ascii=False)
        reader.write(data)


def generate_entity_file(in_file_name, out_file_name):
    """
    Generate entity information of hotpotQA like dataset.
    Using Hanlp for entity recognizing.
    :param in_file_name:
    :param out_file_name:
    """
    with open(in_file_name, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    analyzer = CRFLexicalAnalyzer()

    entity_data = {}
    for data in tqdm(full_data):
        key = data['_id']
        paras = data['context']
        entity_data[key] = {}
        for para in paras:
            entities = []
            title = para[0]
            sents = para[1]
            context = "".join(sents)
            words = analyzer.analyze(context)
            for word in words:
                if word.label.startswith("n") and str(word.getValue())not in entities:
                    entities.append((str(word.getValue()), "1"))
            entity_data[key][title] = entities

    with open(out_file_name, 'w', encoding='utf-8') as reader:
        data = json.dumps(entity_data, indent=4, ensure_ascii=False)
        reader.write(data)


def generate_para_file(in_file_name, out_file_name):
    """
    Generate paragraph information of a hotpotQA like dataset file.
    :param in_file_name:
    :param out_file_name:
    """
    with open(in_file_name, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    paras_data = {}
    for data in full_data:
        key = data['_id']
        paras = data['context']
        paras_data[key] = paras

    with open(out_file_name, 'w', encoding='utf-8') as reader:
        data = json.dumps(paras_data, indent=4, ensure_ascii=False)
        reader.write(data)


def generate_graph_file(feature_file, example_file, entity_file, output_file):
    with gzip.open(example_file, 'rb') as fin:
        examples = pickle.load(fin)
        example_dict = {e.qas_id: e for e in examples}

    with gzip.open(feature_file, 'rb') as fin:
        features = pickle.load(fin)

    with open(entity_file, 'r', encoding='utf-8') as fin:
        query_entities = json.load(fin)

    entity_cnt = []
    entity_graphs = {}
    for case in tqdm(features):
        case.__dict__['answer'] = example_dict[case.qas_id].orig_answer_text
        case.__dict__['query_entities'] = [ent[0] for ent in query_entities[case.qas_id]]
        graph = create_entity_graph(case, 80, 512, 'sent', False, False, relational=False)
        entity_cnt.append(graph['entity_length'])

        # Simplify Graph dicts
        targets = ['entity_length', 'start_entities', 'entity_mapping', 'adj']
        simp_graph = dict([(t, graph[t]) for t in targets])

        entity_graphs[case.qas_id] = simp_graph

    entity_cnt = np.array(entity_cnt)
    for thr in range(40, 100, 10):
        print(len(np.where(entity_cnt > thr)[0]) / len(entity_cnt), f'> {thr}')

    pickle.dump(entity_graphs, gzip.open(output_file, 'wb'))


def sample_dev(full_data_file, train_file_output, dev_file_output, ratio=0.1, seed=None):
    random.seed(seed)

    with open(full_data_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    random.shuffle(full_data)
    n = len(full_data)
    split = int(n * ratio)

    dev_set, train_set = full_data[:split], full_data[split:]

    def id(case):
        return case['_id']

    dev_set.sort(key=id)
    train_set.sort(key=id)

    with open(train_file_output, 'w', encoding='utf-8') as reader:
        data = json.dumps(train_set, indent=4, ensure_ascii=False)
        reader.write(data)

    with open(dev_file_output, 'w', encoding='utf-8') as reader:
        data = json.dumps(dev_set, indent=4, ensure_ascii=False)
        reader.write(data)



