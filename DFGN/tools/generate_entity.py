from pyhanlp import *
import json
import tqdm

if __name__ == "__main__":
    with open("../data/train_name.json", 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    analyzer = CRFLexicalAnalyzer()

    entity_data = {}
    for data in tqdm.tqdm(full_data):
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

    with open("../data/entity.json", 'w', encoding='utf-8') as reader:
        data = json.dumps(entity_data, indent=4, ensure_ascii=False)
        reader.write(data)