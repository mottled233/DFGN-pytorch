import re
import tqdm
import json
import random


#
# with open("data/train1.json", 'w', encoding='utf-8') as reader:
#     data = json.dumps(full_data, indent=4, ensure_ascii=False)
#     reader.write(data)

# from pyhanlp import *
# CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
# analyzer = CRFLexicalAnalyzer()
# print(analyzer.analyze("经审理查明：被告江奕云系北京市东城区街号楼单元号房屋业主，由原告为该房屋提供冬季供暖服务，"))

name_list = ["张吉惟","林国瑞","林玟书","江奕云","刘柏宏","阮建安","王帆","夏志豪","吉茹定","李中冰",
             "黄文隆","谢彦文","傅智翔","洪振霞","刘姿婷","荣姿康","吕致盈","方强","黎贵","郑伊雯","雷宝",
             "吴美隆","王美珠","郭芳","李雅惠","陈文婷","曹敏侑","王依婷","陈婉璇","吴玉","蔡依婷",
             "郑梦","林家纶","黄丽","黄芸欢","吴韵如","李肇芬","卢木仲","李成白","方兆玉","刘翊惠",
             "丁汉臻","吴佳瑞","舒绿珮","周白芷","张姿妤","张虹伦","周琼玟","倪怡芳","杨佩芳","黄文旺",
             "黄盛玫","郑丽青","许智云","张孟涵","李爱","王恩龙","朱政廷","邓诗涵","陈政倩","吴俊伯","阮馨学",
             "翁惠珠","吴思翰","林佩玲","邓海","陈翊依","李建智","武淑芬","金雅琪","赖怡宜","黄育霖","张仪",
             "王俊民","张诗刚","林慧颖","沈俊君","陈淑妤","李姿伶","高咏钰","黄彦宜","周孟儒","潘欣臻","李祯韵"]

patterns = [re.compile(r"[\u4e00-\u9fa5][xX×]+\d{1,2}"),
            re.compile(r"[\u4e00-\u9fa5]某[\u4e00-\u9fa5]?")]


def get_random_name(name_to_ori={}):
    while True:
        idx = random.randint(0, len(name_list)-1)
        name = name_list[idx]
        if name not in name_to_ori:
            return name


def trans_name(example):
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


if __name__ == "__main__":
    # clean the information desensitization
    with open("../data/train.json", 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    for data in tqdm.tqdm(full_data):
        trans_name(data)

    with open("../data/train_name.json", 'w', encoding='utf-8') as reader:
        data = json.dumps(full_data, indent=4, ensure_ascii=False)
        reader.write(data)


