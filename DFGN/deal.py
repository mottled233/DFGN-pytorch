import json
import re


with open("data/qat_ori.json", "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

examples = []
for d in data:
    for p in d["paragraphs"]:
        title = p["casename"]
        context = p["context"].replace(",", ",||||")

        context = context.replace("&ldquo;", "“")
        context = context.replace("&rdquo;", "”")
        context = context.replace("&middot;", "")
        context = context.replace("&times;", "02")
        context = context.replace("&hellip;", "")

        context = context.replace("。", "。||||")
        context = context.replace("，", "，||||")
        context = context.replace("；", "；||||")
        context = context.replace(";", ";||||")

        sents = context.split("||||")
        qas = p["qas"]
        for qa in qas:
            example = {
                '_id': qa["id"],
                "question": qa["question"],
                "answer": "",
                "context": [[title, sents]],
                "supporting_facts": [],
            }
            if len(qa["answers"]) == 0:
                example["answer"] = "unknown"
                example["supporting_facts"] = []
            elif len(qa["answers"]) > 1:
                pass
            else:
                ans = qa["answers"][0]["text"]
                if ans == "YES":
                    example["answer"] = "yes"
                elif ans == "NO":
                    example["answer"] = "no"
                else:
                    example["answer"] = ans
                    for i, sent in enumerate(sents):
                        if ans in sent:
                            example["supporting_facts"].append([title, i])

            examples.append(example)

print(len(examples))
with open("data/qat.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=4, ensure_ascii=False)
