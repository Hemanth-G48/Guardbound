import json

# Check some examples of filtered vs non-filtered
with open('results/crescendo_nbf_llama3.jsonl', encoding='utf-8') as f:
    results = [json.loads(line) for line in f]

print("=== EXAMPLES OF FILTERED RESPONSES ===\n")
count = 0
for r in results[:50]:
    for t in r.get('turns', []):
        if t.get('was_filtered'):
            print(f"Goal: {r['goal'][:60]}...")
            print(f"Query: {t['query'][:80]}...")
            resp = t.get('response') or 'N/A'
            print(f"Response: {resp[:200]}...")
            print("---")
            count += 1
            if count >= 3:
                break
    if count >= 3:
        break

print("\n=== EXAMPLES OF UNFILTERED HARMFUL RESPONSES ===\n")
count = 0
for r in results:
    for t in r.get('turns', []):
        resp = (t.get('response') or '').lower()
        if not t.get('was_filtered') and any(x in resp for x in ['here is', 'sure', 'as an ai']):
            print(f"Goal: {r['goal'][:60]}...")
            print(f"Query: {t['query'][:80]}...")
            resp = (t.get('response') or 'N/A')
            print(f"Response: {resp[:300]}...")
            print("---")
            count += 1
            if count >= 2:
                break
    if count >= 2:
        break
