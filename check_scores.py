import json

scores = set()
with open('data/processed/conversations/gpt35_circuit_breakers.jsonl', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        for t in d['turns']:
            js = t.get('judge_score')
            if js is not None:
                scores.add(js)

print(f"Non-null judge_scores found: {scores}")
print(f"Total unique: {len(scores)}")
