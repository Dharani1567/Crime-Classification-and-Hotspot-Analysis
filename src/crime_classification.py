"""
src/crime_classification.py
Part 3: Crime Category Classification Across Four Regions

Trains and evaluates 5 machine learning models independently on each dataset:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (LinearSVC with CalibratedClassifierCV)
5. XGBoost (Extreme Gradient Boosting)

Evaluation & Metrics:
- Chronological train/test split (80% train, 20% test)
- Training-only preprocessing: Target Encoding / One-Hot Encoding & StandardScaler fit strictly on train split
- Computes: Accuracy, Macro Precision, Macro Recall, Macro-F1, Weighted-F1, Confusion Matrix, Classification Report
- Macro-F1 is the primary comparison metric
- Saves trained model checkpoints, metric JSONs, comparison tables, confusion matrix plots, and feature importances
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

DATASET_CONFIGS = {
    'chicago': {
        'name': 'Chicago',
        'file': 'data/features/chicago/chicago_task1_crime_category.csv',
        'target_col': 'crime_category_clean',
        'cat_cols': ['district', 'ward', 'community_area', 'beat', 'location_description'],
        'num_cols': [
            'hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek',
            'is_weekend', 'month', 'sin_month', 'cos_month', 'latitude', 'longitude',
            'has_valid_coordinates', 'arrest', 'domestic'
        ],
        'top_n_classes': 15 # Train on top 15 categories representing >96% of cases
    },
    'los_angeles': {
        'name': 'Los Angeles',
        'file': 'data/features/los_angeles/los_angeles_task1_crime_category.csv',
        'target_col': 'crime_category_clean',
        'cat_cols': ['area', 'area_name', 'rpt_dist_no', 'premis_desc', 'weapon_desc', 'vict_sex', 'vict_descent', 'status_desc'],
        'num_cols': [
            'hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek',
            'is_weekend', 'month', 'sin_month', 'cos_month', 'latitude', 'longitude',
            'has_valid_coordinates', 'vict_age'
        ],
        'top_n_classes': 15
    },
    'san_francisco': {
        'name': 'San Francisco',
        'file': 'data/features/san_francisco/san_francisco_task1_crime_category.csv',
        'target_col': 'crime_category_clean',
        'cat_cols': ['police_district', 'analysis_neighborhood', 'supervisor_district', 'report_type_description', 'resolution'],
        'num_cols': [
            'hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek',
            'is_weekend', 'month', 'sin_month', 'cos_month', 'latitude', 'longitude',
            'has_valid_coordinates', 'filed_online'
        ],
        'top_n_classes': 15
    },
    'uk': {
        'name': 'UK',
        'file': 'data/features/uk/uk_task1_crime_category.csv',
        'target_col': 'crime_category_clean',
        'cat_cols': ['location_type', 'street_name', 'outcome_category'],
        'num_cols': ['month_num', 'sin_month', 'cos_month', 'latitude', 'longitude', 'has_valid_coordinates'],
        'top_n_classes': 14 # All standard UK Home Office categories
    }
}

def prepare_data(config, sample_size=40000):
    """
    Loads features, performs chronological train/test split (80/20),
    and fits encoders/scalers STRICTLY on training split.
    """
    print(f"\n--- Loading and Splitting Data: {config['name']} ---")
    df = pd.read_csv(config['file'])
    
    # Restrict to top_n_classes for consistent multi-class comparison across models
    target_counts = df[config['target_col']].value_counts()
    valid_targets = target_counts.head(config['top_n_classes']).index.tolist()
    df = df[df[config['target_col']].isin(valid_targets)].reset_index(drop=True)
    
    # Subsample if dataset is large, preserving chronological order
    if len(df) > sample_size:
        step = len(df) // sample_size
        df = df.iloc[::step].head(sample_size).reset_index(drop=True)
        
    print(f"Working dataset size: {len(df):,} records, {config['top_n_classes']} crime classes.")

    # Chronological Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Encode Target
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[config['target_col']])
    
    # Handle any unseen target in test (map to nearest or filter)
    test_mask = test_df[config['target_col']].isin(label_encoder.classes_)
    test_df = test_df[test_mask].reset_index(drop=True)
    y_test = label_encoder.transform(test_df[config['target_col']])

    # Feature Matrix Construction: Categorical Frequency/Target Encoding on Train
    X_train_num = train_df[config['num_cols']].fillna(0).values
    X_test_num = test_df[config['num_cols']].fillna(0).values

    # Scale numeric features strictly fit on Train
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # Categorical Frequency Encoding (fit on Train only)
    X_train_cat_list = []
    X_test_cat_list = []
    cat_feature_names = []
    
    for c in config['cat_cols']:
        freq_map = train_df[c].astype(str).value_counts(normalize=True).to_dict()
        train_enc = train_df[c].astype(str).map(freq_map).fillna(0).values.reshape(-1, 1)
        test_enc = test_df[c].astype(str).map(freq_map).fillna(0).values.reshape(-1, 1)
        X_train_cat_list.append(train_enc)
        X_test_cat_list.append(test_enc)
        cat_feature_names.append(f"{c}_freq")

    X_train_cat = np.hstack(X_train_cat_list)
    X_test_cat = np.hstack(X_test_cat_list)

    # Combine Numeric Scaled + Categorical Encoded
    X_train = np.hstack([X_train_num_scaled, X_train_cat])
    X_test = np.hstack([X_test_num_scaled, X_test_cat])
    
    feature_names = config['num_cols'] + cat_feature_names

    return X_train, X_test, y_train, y_test, label_encoder, feature_names

def train_and_evaluate_models(region_key, config):
    print(f"\n==================================================")
    print(f"TRAINING CRIME CLASSIFICATION MODELS: {config['name'].upper()}")
    print(f"==================================================")
    
    X_train, X_test, y_train, y_test, label_encoder, feature_names = prepare_data(config, sample_size=40000)
    classes = label_encoder.classes_
    num_classes = len(classes)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500,
            C=1.0,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1
        ),
        'Support Vector Machine': CalibratedClassifierCV(
            LinearSVC(C=0.5, random_state=42, max_iter=2000),
            cv=3
        ),
        'XGBoost': XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }

    metrics_list = []
    
    model_dir = f"models/crime_classification/{region_key}"
    metric_dir = f"results/metrics/crime_classification/{region_key}"
    figure_dir = f"results/figures/crime_classification/{region_key}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n>> Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"   Accuracy:    {acc:.4f}")
        print(f"   Macro Prec:  {macro_prec:.4f}")
        print(f"   Macro Rec:   {macro_rec:.4f}")
        print(f"   Macro-F1:    {macro_f1:.4f} (Primary)")
        print(f"   Weighted-F1: {weighted_f1:.4f}")

        # Save checkpoint
        safe_name = name.lower().replace(' ', '_')
        model_path = f"{model_dir}/{safe_name}.joblib"
        joblib.dump(model, model_path)

        # Classification report
        cls_report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True, zero_division=0)
        with open(f"{metric_dir}/{safe_name}_report.json", 'w', encoding='utf-8') as f:
            json.dump(cls_report, f, indent=2)

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name} ({config['name']})\nMacro-F1: {macro_f1:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        cm_path = f"{figure_dir}/{safe_name}_cm.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()

        # Feature Importance (for RF and XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            feat_df.to_csv(f"{metric_dir}/{safe_name}_feature_importance.csv", index=False)
            
            plt.figure(figsize=(9, 6))
            sns.barplot(data=feat_df.head(12), x='importance', y='feature', palette='viridis')
            plt.title(f"Top 12 Features: {name} ({config['name']})")
            plt.xlabel("Relative Importance")
            plt.tight_layout()
            plt.savefig(f"{figure_dir}/{safe_name}_feature_importance.png", dpi=200)
            plt.close()

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Macro Precision': round(macro_prec, 4),
            'Macro Recall': round(macro_rec, 4),
            'Macro-F1': round(macro_f1, 4),
            'Weighted-F1': round(weighted_f1, 4)
        })

    # Summary table per dataset
    results_df = pd.DataFrame(metrics_list)
    results_csv = f"{metric_dir}/model_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved Comparison Table to {results_csv}")
    print(results_df.to_markdown(index=False))

    # Comparison Bar Plot
    plt.figure(figsize=(9, 5))
    df_plot = results_df.melt(id_vars='Model', value_vars=['Macro-F1', 'Weighted-F1', 'Accuracy'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title(f"Model Performance Comparison: {config['name']} Crime Classification")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    comparison_plot_path = f"{figure_dir}/model_comparison_plot.png"
    plt.savefig(comparison_plot_path, dpi=200)
    plt.close()

    return results_df

def run_all_classification():
    all_results = {}
    for region_key, config in DATASET_CONFIGS.items():
        res_df = train_and_evaluate_models(region_key, config)
        all_results[config['name']] = res_df.to_dict(orient='records')

    with open('results/metrics/crime_classification/all_regions_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print("\nALL FOUR REGIONS TRAINED AND EVALUATED FOR TASK 1.")

if __name__ == '__main__':
    run_all_classification()
