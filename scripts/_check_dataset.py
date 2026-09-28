import json
d = json.load(open('nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json'))
print(f'Goals: {len(d)}')
print(f'Keys: {list(d[0].keys())}')
print(f'First goal: {d[0].get("task", "?")[:100]}')
