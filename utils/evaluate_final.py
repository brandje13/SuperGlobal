from typing import Set, Any
import numpy as np


def evaluate_final(cfg, models, results, mode):
    print(f"Evaluating final results with mode: {mode}")

    # --- Helper: Smart Path Cleaner ---
    def clean_path(p):
        # 1. Normalize slashes to backslashes (Windows style)
        p = p.replace('/', '\\')

        # 2. Strip everything before "queries\" to match Ground Truth format
        # Input: ".\datasets\ILIAS_Test\queries\bold_bimp...\img.jpg"
        # Output: "queries\bold_bimp...\img.jpg"
        keyword = "queries\\"
        if keyword in p:
            start_index = p.find(keyword)
            return p[start_index:]

        # Fallback: if 'queries\' isn't found, try to strip the root dir
        root = cfg['dir_data'].replace('/', '\\')
        if p.startswith(root):
            p = p.replace(root, '')

        # Remove leading dot-slash or slash if present
        if p.startswith('.\\'): p = p[2:]
        if p.startswith('\\'): p = p[1:]

        return p

    # ----------------------------------

    # 1. Evaluate Individual Models
    for model_entry in models:
        # Unpack the list/tuple: ['Name', PredictionsDict]
        model_name = model_entry[0]
        preds_dict = model_entry[1]

        precisions = []
        recalls = []
        f1_scores = []
        f2_scores = []
        f3_scores = []
        expected_list = []

        # Iterate via Index (i) to align QIMLIST with GND LIST
        for i, query_name in enumerate(cfg['qimlist']):
            # A. Get Predictions & Clean Paths
            raw_preds = preds_dict.get(query_name)['top_k']
            predicted = set([clean_path(p) for p in raw_preds])

            # B. Get Ground Truth
            gnd_item = cfg['gnd'][i]
            expected = set(gnd_item['ok']) | set(gnd_item['good'])
            expected_list.append(len(expected))

            # C. Calculate Metrics
            p = precision(predicted, expected)
            r = recall(predicted, expected)
            f1 = f_beta(p, r, beta=1.0)  # Standard F1 Score
            f2 = f_beta(p, r, beta=2.0)
            f3 = f_beta(p, r, beta=3.0)

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            f2_scores.append(f2)
            f3_scores.append(f3)

        print(f"[{model_name}]")
        print(f"  Precision: {np.mean(precisions):.4f}")
        print(f"  Recall:    {np.mean(recalls):.4f}")
        print(f"  F1 Score:  {np.mean(f1_scores):.4f}")
        print(f"  F2 Score:  {np.mean(f2_scores):.4f}")
        print(f"  F3 Score:  {np.mean(f3_scores):.4f}")
        print("-" * 30)
        print("  Expected Counts per Query: ", max(expected_list), " (Max), ", min(expected_list), " (Min), ", np.mean(expected_list), " (Avg)")
        print(expected_list)

    # 2. Evaluate Multiview Results
    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    f3_scores = []

    for i, query_name in enumerate(cfg['qimlist']):
        # A. Get Results
        raw_results = results.get(query_name, [])
        predicted = set([clean_path(p) for p in raw_results])

        # B. Get Ground Truth
        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])

        p = precision(predicted, expected)
        r = recall(predicted, expected)
        f1 = f_beta(p, r, beta=1.0)  # Standard F1 Score
        f2 = f_beta(p, r, beta=2.0)
        f3 = f_beta(p, r, beta=3.0)

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        f2_scores.append(f2)
        f3_scores.append(f3)

    print(f"[Multiview Results]")
    print(f"  Precision: {np.mean(precisions):.4f}")
    print(f"  Recall:    {np.mean(recalls):.4f}")
    print(f"  F1 Score:  {np.mean(f1_scores):.4f}")
    print(f"  F2 Score:  {np.mean(f2_scores):.4f}")
    print(f"  F3 Score:  {np.mean(f3_scores):.4f}")


# --- Metrics Helper Functions ---
def precision(predicted: Set[Any], expected: Set[Any]) -> float:
    if not predicted: return 0.0
    return len(predicted & expected) / len(predicted)


def recall(predicted: Set[Any], expected: Set[Any]) -> float:
    if not expected: return 0.0
    return len(predicted & expected) / len(expected)


def f_beta(p: float, r: float, beta=2.0) -> float:
    """
    Calculates the F-Beta score.
    beta=2 weights recall twice as much as precision.
    beta=3 weights recall three times as much as precision.
    """
    if p == 0 and r == 0:
        return 0.0

    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)

    return f_beta