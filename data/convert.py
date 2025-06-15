import json

input_path = "npc_dialogues.jsonl"
output_path = "finetune_data.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue  # 空行をスキップ
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"スキップ: {line}")
            continue
        prompt = " ".join(item["history"])
        response = item["response"]
        fout.write(json.dumps({"text": f"{prompt}\n{response}"}, ensure_ascii=False) + "\n")