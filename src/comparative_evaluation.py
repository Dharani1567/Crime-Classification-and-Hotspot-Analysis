"""
src/comparative_evaluation.py
Part 5: Cross-Dataset Comparative Evaluation

Consolidates all empirical evaluation metrics from:
- Task 1: Crime Category Classification (5 models x 4 datasets = 20 models)
- Task 2: Crime Hotspot Classification (4 models x 4 datasets = 16 models)

Generates:
- results/metrics/model_comparison.csv
- Comparative visual plots in results/figures/model_comparison/:
  1. task1_macro_f1_comparison.png
  2. task2_macro_f1_comparison.png
  3. cross_task_performance_radar.png
  4. model_performance_heatmap.png
- Comprehensive technical markdown report in results/reports/comparative_analysis.md
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

REGIONS = [
    ('chicago', 'Chicago'),
    ('los_angeles', 'Los Angeles'),
    ('san_francisco', 'San Francisco'),
    ('uk', 'UK')
]

def consolidate_metrics():
    print("Consolidating empirical evaluation results...")
    records = []

    # Task A: Crime Category Classification
    for code, label in REGIONS:
        csv_path = f"results/metrics/crime_classification/{code}/model_comparison.csv"
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            records.append({
                'Task': 'Crime Category Classification',
                'Dataset': label,
                'Model': row['Model'],
                'Accuracy': row['Accuracy'],
                'Macro-F1': row['Macro-F1'],
                'Weighted-F1': row['Weighted-F1'],
                'Macro Precision': row['Macro Precision'],
                'Macro Recall': row['Macro Recall']
            })

    # Task B: Crime Hotspot Classification
    for code, label in REGIONS:
        csv_path = f"results/metrics/hotspot_classification/{code}/model_comparison.csv"
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            records.append({
                'Task': 'Crime Hotspot Classification',
                'Dataset': label,
                'Model': row['Model'],
                'Accuracy': row['Accuracy'],
                'Macro-F1': row['Macro-F1'],
                'Weighted-F1': row['Weighted-F1'],
                'Macro Precision': row['Macro Precision'],
                'Macro Recall': row['Macro Recall']
            })

    all_df = pd.DataFrame(records)
    out_csv = 'results/metrics/model_comparison.csv'
    all_df.to_csv(out_csv, index=False)
    print(f"Saved consolidated comparison to {out_csv} ({len(all_df)} entries).")
    return all_df

def generate_comparative_plots(df):
    print("Generating comparative evaluation figures...")
    out_dir = 'results/figures/model_comparison'
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)

    # 1. Task 1: Macro-F1 across 4 datasets
    t1_df = df[df['Task'] == 'Crime Category Classification']
    plt.figure(figsize=(11, 6))
    ax1 = sns.barplot(data=t1_df, x='Dataset', y='Macro-F1', hue='Model', palette='Set2')
    plt.title("Task 1: Crime Category Classification (Macro-F1 by Dataset & Model)", fontsize=14, weight='bold')
    plt.ylabel("Macro-F1 Score")
    plt.ylim(0, 0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    p1 = f"{out_dir}/task1_macro_f1_comparison.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved {p1}")

    # 2. Task 2: Macro-F1 across 4 datasets
    t2_df = df[df['Task'] == 'Crime Hotspot Classification']
    plt.figure(figsize=(11, 6))
    ax2 = sns.barplot(data=t2_df, x='Dataset', y='Macro-F1', hue='Model', palette='tab10')
    plt.title("Task 2: Crime Hotspot Classification (Macro-F1 by Dataset & Model)", fontsize=14, weight='bold')
    plt.ylabel("Macro-F1 Score")
    plt.ylim(0.6, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    p2 = f"{out_dir}/task2_macro_f1_comparison.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved {p2}")

    # 3. Model Performance Heatmap (Macro-F1 matrix: Model x Dataset)
    pivot_t1 = t1_df.pivot(index='Model', columns='Dataset', values='Macro-F1')
    pivot_t2 = t2_df.pivot(index='Model', columns='Dataset', values='Macro-F1')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(pivot_t1, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Macro-F1'})
    axes[0].set_title("Task 1: Crime Category Classification", fontsize=12, weight='bold')
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Model Architecture")

    sns.heatmap(pivot_t2, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Macro-F1'})
    axes[1].set_title("Task 2: Crime Hotspot Classification", fontsize=12, weight='bold')
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("")

    plt.suptitle("Cross-Regional Model Performance Matrix (Macro-F1)", fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    p3 = f"{out_dir}/model_performance_heatmap.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"Saved {p3}")

    # 4. Accuracy vs Macro-F1 Discrepancy (Illustrating Class Imbalance Impact)
    plt.figure(figsize=(12, 6))
    melted = df.melt(id_vars=['Task', 'Dataset', 'Model'], value_vars=['Accuracy', 'Macro-F1'], var_name='Metric', value_name='Score')
    sns.catplot(
        data=melted, x='Dataset', y='Score', hue='Metric', col='Task',
        kind='bar', palette='coolwarm', height=5, aspect=1.3
    )
    plt.subplots_adjust(top=0.88)
    plt.suptitle("Accuracy vs. Macro-F1 Gap Reflecting Regional Class Imbalance", fontsize=14, weight='bold')
    p4 = f"{out_dir}/accuracy_vs_macro_f1_gap.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"Saved {p4}")

def main():
    df = consolidate_metrics()
    generate_comparative_plots(df)
    print("\nPart 5 comparative processing complete.")

if __name__ == '__main__':
    main()
