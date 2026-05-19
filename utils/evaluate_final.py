from typing import Set, Any, Dict
import numpy as np
import os


def evaluate_final(cfg, models, results, mode, silent=False):
    if not silent:
        print(f"\nEvaluating final results with mode: {mode}")

    # --- Helper: Smart Path Cleaner ---
    def clean_path(p):
        # Normalize path to the current OS
        p = os.path.normpath(p)
        keyword = "queries"

        # Check for keyword (avoids manual slash handling)
        if keyword in p.split(os.sep):
            return os.path.join(keyword, p.split(keyword)[-1].lstrip(os.sep))

        root = os.path.normpath(cfg['dir_data'])

        # Remove root path safely
        if p.startswith(root):
            p = p[len(root):]

        return p.lstrip(os.sep)

    # 1. Evaluate Individual Models (Only if not silent)
    if not silent:
        for model_entry in models:
            model_name = model_entry[0]
            preds_dict = model_entry[1]

            precisions, recalls, f1_scores = [], [], []
            expected_list = []

            for i, query_name in enumerate(cfg['qimlist']):
                raw_preds = preds_dict.get(query_name)['top_k']
                predicted = set([clean_path(p) for p in raw_preds])
                gnd_item = cfg['gnd'][i]
                expected = set(gnd_item['ok']) | set(gnd_item['good'])
                expected_list.append(len(expected))

                p = precision(predicted, expected)
                r = recall(predicted, expected)
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f_beta(p, r, beta=1.0))

            print(f"[{model_name}]")
            print(f"  Precision: {np.mean(precisions):.4f}")
            print(f"  Recall:    {np.mean(recalls):.4f}")
            print(f"  F1 Score:  {np.mean(f1_scores):.4f}")
            print("-" * 30)

    # 2. Evaluate Multiview Results (Always calculated, only printed if not silent)
    m_precisions, m_recalls = [], []
    m_f1, m_f2, m_f3 = [], [], []

    for i, query_name in enumerate(cfg['qimlist']):
        raw_results = results.get(query_name, [])
        predicted = set([clean_path(p) for p in raw_results])
        gnd_item = cfg['gnd'][i]
        expected = set(gnd_item['ok']) | set(gnd_item['good'])

        p = precision(predicted, expected)
        r = recall(predicted, expected)

        m_precisions.append(p)
        m_recalls.append(r)
        m_f1.append(f_beta(p, r, beta=1.0))
        m_f2.append(f_beta(p, r, beta=2.0))
        m_f3.append(f_beta(p, r, beta=3.0))

    avg_metrics = {
        "precision": float(np.mean(m_precisions)),
        "recall": float(np.mean(m_recalls)),
        "f1": float(np.mean(m_f1)),
        "f2": float(np.mean(m_f2)),
        "f3": float(np.mean(m_f3))
    }

    if not silent:
        print(f"[Multiview Results - {mode}]")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")
        print(f"  F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"  F3 Score:  {avg_metrics['f3']:.4f}")

    return avg_metrics


# --- Metrics Helper Functions ---
def precision(predicted: Set[Any], expected: Set[Any]) -> float:
    if not predicted: return 0.0
    return len(predicted & expected) / len(predicted)


def recall(predicted: Set[Any], expected: Set[Any]) -> float:
    if not expected: return 0.0
    return len(predicted & expected) / len(expected)


def f_beta(p: float, r: float, beta=2.0) -> float:
    if p == 0 and r == 0: return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)