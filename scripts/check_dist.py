import json

# Check the saved distribution info
dist_path = 'data/processed/datasets/nbf_pseudo_labels/scoring_summary.json'
with open(dist_path) as f:
    dist = json.load(f)
print('Original scoring summary:')
for k, v in dist['score_distribution'].items():
    pct = dist['percentages'][k]
    print(f'  Score {k}: {v:6d} ({pct})')

# Now check what was built
built_dist_path = 'data/processed/datasets/nbf_labeled/label_distribution.json'
with open(built_dist_path) as f:
    built = json.load(f)
print('\nBuilt dataset distribution:')
for k, v in built['score_distribution'].items():
    pct = built['percentages'][k]
    print(f'  Score {k}: {v:6d} ({pct})')

# Total turns
print(f'\nOriginal total turns: {dist["total_turns"]}')
print(f'Built total turns: {built["total_turns"]}')

# Check the mismatch
orig = dist['score_distribution']
built_s = built['score_distribution']
print('\nMismatch analysis:')
for k in orig:
    diff = orig[k] - built_s.get(k, 0)
    print(f'  Score {k}: orig={orig[k]}, built={built_s.get(k, 0)}, diff={diff}')
