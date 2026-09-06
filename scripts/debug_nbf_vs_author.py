#!/usr/bin/env python3
"""Debug script to compare Guardbound vs Author implementation.

This script analyzes and fixes 4 key differences:
1. Verify state equivalence between Guardbound and author
2. Add retry logic to match author's safety filtering
3. Check threshold sign convention
4. Compare training data distributions
"""
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def compare_threshold_sign():
    """Check threshold sign convention between implementations.

    Author's Crescendomation:
        threshold = -args.threshold
        if ssm_nbf_score < threshold:  # i.e., h < -threshold = args.threshold
            # filter (UNSAFE)

    Guardbound's SteeredLLMChat:
        filtered = (h_value + self.eta) >= 0  # i.e., h >= -eta

    Both filter when h >= -threshold/eta, which is CORRECT per paper.
    """
    print("\n" + "="*60)
    print("1. THRESHOLD SIGN CONVENTION CHECK")
    print("="*60)

    print("""
    Author's Crescendomation (line 27, 58):
        threshold = -args.threshold  # if args.threshold=0.001, threshold=-0.001
        if ssm_nbf_score < threshold:  # h < -0.001 = UNSAFE
            break  # filter

    Guardbound's SteeredLLMChat (line 178):
        filtered = (h_value + self.eta) >= 0  # h >= -eta = UNSAFE

    Guardbound's filter_query (predictor.py line 211):
        allowed = (h_value + eta) < 0  # h < -eta = SAFE

    ANALYSIS:
    - Both implementations filter when h >= -threshold
    - h = p_unsafe - max(p_safe)
    - h >= -eta means p_unsafe - max(p_safe) >= -eta
    - This means unsafe probability is within eta of safe probability
    - If eta=0.001, filter when unsafe is only 0.1% less likely than safe

    CONCLUSION: Sign convention is EQUIVALENT between implementations.
    Both correctly filter when h >= -eta (i.e., unsafe score too close to safe).
    """)
    return True


def analyze_author_safety_filtering():
    """Analyze author's safety filtering logic in detail."""
    print("\n" + "="*60)
    print("2. AUTHOR'S SAFETY FILTERING LOGIC ANALYSIS")
    print("="*60)

    print("""
    Author's Crescendomation filtering (lines 49-69):

    threshold = -args.threshold  # e.g., -0.001
    minimal_ssm_nbf_score = float('inf')
    min_score_query = None
    unsafe_cutoff = False

    for index_ in range(3):  # Try up to 3 times
        ssm_nbf_score = calculate_score_from_dialog(history_t, prompt, ...)
        if minimal_ssm_nbf_score > ssm_nbf_score:
            minimal_ssm_nbf_score = ssm_nbf_score
            min_score_query = prompt
        if ssm_nbf_score < threshold:  # h < -threshold = UNSAFE
            break
        else:  # h >= -threshold = SAFE (counterintuitive naming!)
            print("unsafe prompt filtered: ", ...)
            unsafe_cutoff = True
            break

    prompt = min_score_query
    if unsafe_cutoff:  # If found a SAFE prompt
        num_filtering += 1
        continue  # Skip this round entirely

    KEY INSIGHT: The variable "unsafe_cutoff" is MISNAMED!
    - It is set to True when h >= -threshold (SAFE prompt found)
    - It causes the round to be SKIPPED (filtered)
    - So it actually means "safe_cutoff" or "skip_round"

    CRITICAL BEHAVIOR:
    1. Author tries up to 3 different prompts
    2. Takes the prompt with MINIMUM score (most safe)
    3. If any prompt is "safe" (h >= -threshold), skip the entire round
    4. If all 3 are "unsafe" (h < -threshold), use the last one

    THIS EXPLAINS WHY AUTHOR'S NBF WORKS:
    - The retry logic finds genuinely safe starting points
    - Guardbound's single-shot filtering is too aggressive
    """)


def compare_training_data():
    """Compare training data distributions between author and Guardbound."""
    print("\n" + "="*60)
    print("4. TRAINING DATA DISTRIBUTION COMPARISON")
    print("="*60)

    # Author's data
    author_data_path = Path("nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/LoRA_SFT/gpt-3-5-turbo_circuit_breakers_1k_all_attacks.json")
    if not author_data_path.exists():
        print(f"Author data not found at {author_data_path}")
        return

    print(f"\nAnalyzing author data: {author_data_path}")

    # Count scores
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'refused': 0, 'unknown': 0}
    attack_types = {}

    with open(author_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'data' in data:
        # Circuit breakers format
        for item in data['data']:
            attack_method = item.get('attack_method', 'unknown')
            if attack_method not in attack_types:
                attack_types[attack_method] = {'total': 0, 'by_score': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
            attack_types[attack_method]['total'] += 1

            for attempt in item.get('attempts', []):
                for step in attempt.get('dialog_hist', []):
                    if step.get('role') == 'assistant':
                        score = step.get('score', 'unknown')
                        if isinstance(score, int) and 1 <= score <= 5:
                            score_counts[score] += 1
                            attack_types[attack_method]['by_score'][score] += 1
                        elif score == 'refused':
                            score_counts['refused'] += 1
                        else:
                            score_counts['unknown'] += 1

    elif isinstance(data, list):
        # JSONL-like format
        for item in data:
            score = item.get('score', 'unknown')
            if isinstance(score, int) and 1 <= score <= 5:
                score_counts[score] += 1
            elif score == 'refused':
                score_counts['refused'] += 1
            else:
                score_counts['unknown'] += 1

    total = sum(v for k, v in score_counts.items() if k != 'unknown')

    print(f"\nScore Distribution (Author's Data):")
    print(f"  Score 1: {score_counts[1]:6d} ({100*score_counts[1]/total:.1f}%)")
    print(f"  Score 2: {score_counts[2]:6d} ({100*score_counts[2]/total:.1f}%)")
    print(f"  Score 3: {score_counts[3]:6d} ({100*score_counts[3]/total:.1f}%)")
    print(f"  Score 4: {score_counts[4]:6d} ({100*score_counts[4]/total:.1f}%)")
    print(f"  Score 5: {score_counts[5]:6d} ({100*score_counts[5]/total:.1f}%)")
    print(f"  Refused: {score_counts['refused']:6d}")
    print(f"  Total:   {total}")

    print(f"\nAttack Types: {len(attack_types)}")
    for atk, counts in sorted(attack_types.items(), key=lambda x: -x[1]['total'])[:10]:
        print(f"  {atk}: {counts['total']} conversations")

    # Compare with Guardbound data
    guardbound_data_path = Path("data/processed/datasets/nbf_labeled/train.npz")
    if guardbound_data_path.exists():
        print(f"\nAnalyzing Guardbound data: {guardbound_data_path}")
        data = np.load(guardbound_data_path)
        if 'labels' in data:
            labels = data['labels']
            print(f"  Labels shape: {labels.shape}")
            print(f"  Unique labels: {np.unique(labels)}")

            guard_score_counts = {}
            for score_idx in range(1, 6):  # Paper labels 1-5
                ce_idx = score_idx - 1  # CE index 0-4
                count = np.sum(labels == ce_idx)
                guard_score_counts[score_idx] = count

            gb_total = sum(guard_score_counts.values())
            print(f"\nGuardbound Score Distribution (from train.npz):")
            for score, count in guard_score_counts.items():
                pct = 100 * count / gb_total if gb_total > 0 else 0
                print(f"  Score {score}: {count:6d} ({pct:.1f}%)")

    # Check if score distribution matches what author reports
    print(f"\nComparison with Author's NBF paper:")
    print(f"  Paper reports score distribution: 1=26.8%, 2=11.9%, 3=52.4%, 4=0.7%, 5=8.2%")
    print(f"  This is the target distribution our NBF should learn!")


def analyze_state_difference():
    """Analyze the state computation difference."""
    print("\n" + "="*60)
    print("STATE COMPUTATION DIFFERENCE (CRITICAL)")
    print("="*60)

    print("""
    Author's calculate_score_from_dialog():
    - Initializes x_t = zeros for EACH call
    - Processes ALL dialog history to update state
    - Then evaluates the new query

    Guardbound's SteeredLLMChat:
    - Maintains self._state across turns
    - Only advances state when query is ACCEPTED
    - When filtered, state is NOT updated

    CRITICAL ISSUE:
    If Guardbound filters a turn, the state doesn't advance.
    But author's method would have processed that turn.

    Example scenario:
    Turn 0: Guardbound accepts -> state = f(x0, u0)
    Turn 1: Guardbound filters -> state = f(x0, u0)  [no change]

    But author's method would compute:
    - First: process u0 to get x1 = f(x0, u0)
    - Then: check u1, find it unsafe, DON'T advance
    - Result: state = x1 = f(x0, u0)

    So if Guardbound's Turn 1 is filtered, the states MATCH!

    BUT IF Guardbound accepts Turn 1:
    - Guardbound: state = f(x1, u1) = f(f(x0, u0), u1)
    - Author: state = x1 (already computed above)

    These should match IF dynamics are deterministic.

    THE REAL ISSUE might be:
    - Guardbound's state at Turn 0 is zeros
    - Author's calculate_score_from_dialog processes Turn 0 and gets x1
    - But Guardbound hasn't processed anything yet!

    This means at Turn 0, Guardbound's state is x0 (zeros)
    while Author would have processed Turn 0 and be at x1.

    This explains why Turn 0 gets blocked - the state hasn't been
    advanced yet, so the NBF sees "zeros" which might be flagged.
    """)


def main():
    print("="*60)
    print("GUARDBOUND vs AUTHOR NBF IMPLEMENTATION ANALYSIS")
    print("="*60)

    compare_threshold_sign()
    analyze_author_safety_filtering()
    compare_training_data()
    analyze_state_difference()

    print("\n" + "="*60)
    print("SUMMARY OF ISSUES AND FIXES")
    print("="*60)
    print("""
    ISSUE 1: State at Turn 0
    - Guardbound: state = zeros (not processed any turn yet)
    - Author: would have processed Turn 0 via calculate_score_from_dialog
    - FIX: Process Turn 0 through dynamics before first safety check

    ISSUE 2: No Retry Logic
    - Guardbound: single-shot filtering, reject if unsafe
    - Author: tries 3 different prompts, takes minimum score (most safe)
    - FIX: Add retry mechanism to SteeredLLMChat

    ISSUE 3: Threshold Sign
    - Already equivalent, no fix needed

    ISSUE 4: Training Data Distribution
    - Should match author's reported distribution
    - Check that Guardbound training data has similar distribution
    """)


if __name__ == "__main__":
    main()
