# merge_data.py
import pandas as pd
import numpy as np
import os

print("STEP 1: Merging data...")

# Check that files exist
required_files = ['manual_model_performance.csv', 'huggingface_models_with_proxy.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"Missing files: {missing_files}")
    print("Please first execute:")
    print("1. manual_performance_dataset.py")
    print("2. collect_data.py (corrected)")
    exit(1)

# Load data
print("Loading data...")
collected = pd.read_csv("huggingface_models_with_proxy.csv")
manual = pd.read_csv("manual_model_performance.csv")

print(f"Collected data: {len(collected)} models")
print(f"Manual data: {len(manual)} models")

# Merge datasets
print("Merging datasets...")
merged = pd.merge(
    collected, 
    manual[['model_id', 'performance_value', 'performance_metric', 'explicability_proxy', 'paper', 'notes']], 
    on='model_id', 
    how='left', 
    suffixes=('_collected', '_manual')
)

print(f"Merged dataset: {len(merged)} models")

# Create unified performance variable
def get_best_performance(row):
    # Priority 1: Manual data (more reliable)
    if pd.notna(row.get('performance_value')):
        return row['performance_value']
    
    # Priority 2: Collected data
    perf_cols = [col for col in row.index if col.startswith('perf_')]
    for col in perf_cols:
        if pd.notna(row[col]):
            return row[col]
    
    return np.nan

merged['performance_final'] = merged.apply(get_best_performance, axis=1)

# Create unified explainability variable
def get_explicability_proxy(row):
    if pd.notna(row.get('explicability_proxy')):
        return row['explicability_proxy']
    elif pd.notna(row.get('model_type_proxy')):
        return row['model_type_proxy']
    else:
        return 'unknown'

merged['explicability_final'] = merged.apply(get_explicability_proxy, axis=1)

# Ordinal coding for explainability
explicability_mapping = {
    'lightweight': 1,
    'medium': 2, 
    'complex': 3,
    'unknown': np.nan
}
merged['explicability_ordinal'] = merged['explicability_final'].map(explicability_mapping)

# Save
output_file = "merged_analysis_dataset.csv"
merged.to_csv(output_file, index=False)

print(f"\nMerged dataset saved to '{output_file}'")
print(f"   Models with performance: {merged['performance_final'].notna().sum()}/{len(merged)}")
print(f"   Models with explainability: {merged['explicability_ordinal'].notna().sum()}/{len(merged)}")

# Data overview
print("\nMERGED DATA OVERVIEW:")
print(merged[['model_id', 'performance_final', 'explicability_final', 'explicability_ordinal']].head(10))