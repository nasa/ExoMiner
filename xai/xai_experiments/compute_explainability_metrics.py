#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, hamming_loss, accuracy_score, jaccard_score

# List of the 11 explainability branches
BRANCHES = [
    'stellar', 'dv_tce_fit', 'global_flux', 'flux_trend',
    'local_centroid', 'flux_periodogram', 'local_odd_even',
    'local_flux', 'local_weak_secondary', 'local_unfolded_flux', 'diff_img'
]

# Mapping of human-readable method names to filenames
PRED_FILES = {
    'Attention - Summed Over Queries': 'ranked_predictions_testset_attn-mean-query.csv',
    'SHAP — Input Level Tensors':       'ranked_predictions_testset_shap.csv',
    'SHAP — Feature Extractor':          'ranked_predictions_testset_shap_features.csv',
    'Authorities':                       'ranked_predictions_with_authorities.csv',
    'Hits':                              'ranked_predictions_with_hits.csv',
    'Hubs':                              'ranked_predictions_with_hubs.csv'
}

# Define multilabel classification strategies

def max_jump(branch_vals):
    total = sum(branch_vals.values())
    if total == 0:
        return []
    items = sorted(branch_vals.items(), key=lambda x: x[1], reverse=True)
    values = [v for _, v in items]
    diffs = [values[i] - values[i+1] for i in range(len(values)-1)]
    k = np.argmax(diffs) + 1
    return [items[i][0] for i in range(k)]


def top_k(branch_vals, k=4):
    total = sum(branch_vals.values())
    if total == 0:
        return []
    items = sorted(branch_vals.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in items[:k]]


def threshold(branch_vals, thresh):
    total = sum(branch_vals.values())
    if total == 0:
        return []
    normalized = {b: v/total for b, v in branch_vals.items()}
    selected = [b for b, v in normalized.items() if v > thresh]
    if not selected:
        # If none above threshold, return top-1
        return top_k(branch_vals, 1)
    return selected

CLASS_METHODS = {
    'Max Jump':        max_jump,
    'Top 4':           lambda vals: top_k(vals, 4),
    'Threshold 0.09':  lambda vals: threshold(vals, 0.09),
    'Threshold 0.12':  lambda vals: threshold(vals, 0.12),
    'Threshold 0.15':  lambda vals: threshold(vals, 0.15)
}

# For NTP cases
NTP_GOLD = ['global_flux', 'local_flux', 'local_unfolded_flux']


def load_gold(gold_csv):
    df = pd.read_csv(gold_csv)
    # Handle potential typo in column name
    if 'explainabilty' in df.columns and 'explainability' not in df.columns:
        df = df.rename(columns={'explainabilty': 'explainability'})
    gold_map = {}
    for _, row in df.iterrows():
        uid = row['uid']
        expl = row.get('explainability', None)
        if pd.isna(expl):
            continue
        labels = [e.strip() for e in expl.split(';') if e.strip()]
        gold_map[uid] = labels
    return gold_map


def parse_args():
    p = argparse.ArgumentParser(description="Explainability methods comparison and metrics")
    p.add_argument('--pred_dir',   required=True, help='Directory with the six prediction CSVs')
    p.add_argument('--gold_csv',   required=True, help='Path to 2min_explainability.csv')
    p.add_argument('--output_dir', default='output', help='Where to save plots')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    gold_map = load_gold(args.gold_csv)

    # Containers for metrics
    at_least_one = pd.DataFrame(index=CLASS_METHODS.keys(), columns=PRED_FILES.keys(), dtype=float)
    all_features = pd.DataFrame(index=CLASS_METHODS.keys(), columns=PRED_FILES.keys(), dtype=float)

    # Hold lists of labels for further metrics
    true_labels = {cm: {pm: [] for pm in PRED_FILES} for cm in CLASS_METHODS}
    pred_labels = {cm: {pm: [] for pm in PRED_FILES} for cm in CLASS_METHODS}

    # Process each explanation method
    for pm_name, fname in PRED_FILES.items():
        df = pd.read_csv(os.path.join(args.pred_dir, fname))
        for cm_name, cm_func in CLASS_METHODS.items():
            for _, row in df.iterrows():
                uid = row['uid']
                lab = row['label']
                # Determine gold labels
                if lab == 'NTP':
                    gold = NTP_GOLD
                else:
                    if uid not in gold_map:
                        continue
                    gold = gold_map[uid]
                # Extract branch scores and predict
                vals = {b: row[b] for b in BRANCHES}
                pred = cm_func(vals)

                true_labels[cm_name][pm_name].append(gold)
                pred_labels[cm_name][pm_name].append(pred)

            # Compute match rates
            y_true = true_labels[cm_name][pm_name]
            y_pred = pred_labels[cm_name][pm_name]
            at_least_one.loc[cm_name, pm_name] = np.mean([bool(set(pred)&set(gt)) for pred, gt in zip(y_pred, y_true)])
            all_features.loc[cm_name, pm_name] = np.mean([set(gt).issubset(pred) for pred, gt in zip(y_pred, y_true)])

    # Compute detailed metrics
    metrics = {}
    mlb = MultiLabelBinarizer(classes=BRANCHES)
    for cm_name in CLASS_METHODS:
        metrics[cm_name] = {}
        for pm_name in PRED_FILES:
            y_true = true_labels[cm_name][pm_name]
            y_pred = pred_labels[cm_name][pm_name]
            Y_true = mlb.fit_transform(y_true)
            Y_pred = mlb.transform(y_pred)

            precisions = precision_score(Y_true, Y_pred, average=None, zero_division=0)
            recalls    = recall_score(Y_true, Y_pred, average=None, zero_division=0)
            micro_p    = precision_score(Y_true, Y_pred, average='micro', zero_division=0)
            micro_r    = recall_score(Y_true, Y_pred, average='micro', zero_division=0)
            macro_p    = precision_score(Y_true, Y_pred, average='macro', zero_division=0)
            macro_r    = recall_score(Y_true, Y_pred, average='macro', zero_division=0)
            ham        = hamming_loss(Y_true, Y_pred)
            exact      = accuracy_score(Y_true, Y_pred)
            jac        = jaccard_score(Y_true, Y_pred, average='samples', zero_division=0)

            metrics[cm_name][pm_name] = {
                'precision_per_branch': dict(zip(BRANCHES, precisions)),
                'recall_per_branch':    dict(zip(BRANCHES, recalls)),
                'micro_precision':      micro_p,
                'micro_recall':         micro_r,
                'macro_precision':      macro_p,
                'macro_recall':         macro_r,
                'hamming_loss':         ham,
                'exact_match_ratio':    exact,
                'jaccard_index':        jac
            }

    # Plotting
    import seaborn as sns

    # Heatmap: At Least One Feature Match
    plt.figure(figsize=(10,6))
    sns.heatmap(at_least_one, annot=True, fmt='.2f')
    plt.title('At Least One Feature Match')
    plt.savefig(os.path.join(args.output_dir, 'heatmap_at_least_one.png'), bbox_inches='tight')
    plt.close()

    # Heatmap: All Gold Features Matched
    plt.figure(figsize=(10,6))
    sns.heatmap(all_features, annot=True, fmt='.2f')
    plt.title('All Gold Features Matched')
    plt.savefig(os.path.join(args.output_dir, 'heatmap_all_features.png'), bbox_inches='tight')
    plt.close()

    # Precision/Recall per branch
    for cm_name in CLASS_METHODS:
        df_p = pd.DataFrame({pm: metrics[cm_name][pm]['precision_per_branch'] for pm in PRED_FILES})
        plt.figure(figsize=(12,8))
        sns.heatmap(df_p, annot=True, fmt='.2f')
        plt.title(f'Precision per Branch - {cm_name}')
        plt.savefig(os.path.join(args.output_dir, f'precision_{cm_name.replace(" ", "_")}.png'), bbox_inches='tight')
        plt.close()

        df_r = pd.DataFrame({pm: metrics[cm_name][pm]['recall_per_branch'] for pm in PRED_FILES})
        plt.figure(figsize=(12,8))
        sns.heatmap(df_r, annot=True, fmt='.2f')
        plt.title(f'Recall per Branch - {cm_name}')
        plt.savefig(os.path.join(args.output_dir, f'recall_{cm_name.replace(" ", "_")}.png'), bbox_inches='tight')
        plt.close()

    # Aggregate metrics heatmaps
    for metric in ['micro_precision', 'macro_precision', 'micro_recall', 'macro_recall', 'hamming_loss', 'exact_match_ratio', 'jaccard_index']:
        df_m = pd.DataFrame({pm: {cm: metrics[cm][pm][metric] for cm in CLASS_METHODS} for pm in PRED_FILES}).T
        plt.figure(figsize=(10,6))
        sns.heatmap(df_m, annot=True, fmt='.2f')
        plt.title(metric.replace('_', ' ').title())
        plt.savefig(os.path.join(args.output_dir, f'heatmap_{metric}.png'), bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
