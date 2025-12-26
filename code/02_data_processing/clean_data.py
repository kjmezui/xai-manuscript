# clean_data.py
import pandas as pd
import numpy as np
import re

print("="*80)
print("PERFORMANCE DATA CLEANING")
print("="*80)

# Load data
df = pd.read_csv("merged_analysis_dataset.csv")

print(f"Before cleaning: {len(df)} models")
print(f"Models with performance: {df['performance_final'].notna().sum()}")

# Function to clean performance values
def clean_performance_value(value):
    if pd.isna(value):
        return np.nan
    
    # Convert to string if not already
    value_str = str(value).strip()
    
    # 1. Remove percentages
    value_str = value_str.replace('%', '')
    
    # 2. Remove spaces
    value_str = value_str.replace(' ', '')
    
    # 3. Handle cases like "0.98855216855484120.982300884955752298.38"
    # This string appears to contain multiple numbers concatenated
    # Try to extract the first valid number
    matches = re.findall(r'\d+\.\d+', value_str)
    if matches:
        # Take first found number
        try:
            return float(matches[0])
        except:
            pass
    
    # 4. Try direct float conversion
    try:
        return float(value_str)
    except:
        # 5. If failed, try to extract any number
        number_match = re.search(r'\d+\.?\d*', value_str)
        if number_match:
            try:
                return float(number_match.group())
            except:
                return np.nan
    
    return np.nan

# Clean performance_final column
df['performance_final_clean'] = df['performance_final'].apply(clean_performance_value)

# Check other performance columns too
perf_cols = [col for col in df.columns if col.startswith('perf_')]
for col in perf_cols:
    if col in df.columns:
        df[f'{col}_clean'] = df[col].apply(clean_performance_value)

# Calculate improved final performance
def get_best_performance_clean(row):
    # Priority 1: Cleaned performance_final value
    if pd.notna(row.get('performance_final_clean')):
        return row['performance_final_clean']
    
    # Priority 2: Cleaned values from other performance columns
    clean_perf_cols = [col for col in df.columns if col.endswith('_clean') and not col.startswith('performance_final')]
    for col in clean_perf_cols:
        if pd.notna(row[col]):
            return row[col]
    
    # Priority 3: Try to extract performance from notes text
    if pd.notna(row.get('notes')):
        notes_text = str(row['notes'])
        # Look for numbers in notes
        number_matches = re.findall(r'(\d+\.\d+)\s*%?', notes_text)
        if number_matches:
            try:
                # Take the largest number (likely a performance score)
                numbers = [float(match) for match in number_matches]
                if numbers:
                    # Filter plausible numbers (between 0 and 100 for scores)
                    plausible = [n for n in numbers if 0 <= n <= 100]
                    if plausible:
                        return max(plausible)  # Take best performance
            except:
                pass
    
    return np.nan

df['performance_clean'] = df.apply(get_best_performance_clean, axis=1)

# Statistics after cleaning
print(f"\nAfter cleaning:")
print(f"  Models with cleaned performance: {df['performance_clean'].notna().sum()}")
print(f"  Models with explainability: {df['explicability_ordinal'].notna().sum()}")

# Cleaned performance distribution
print(f"\nCleaned performance distribution:")
if df['performance_clean'].notna().sum() > 0:
    print(f"  Min: {df['performance_clean'].min():.2f}")
    print(f"  Max: {df['performance_clean'].max():.2f}")
    print(f"  Mean: {df['performance_clean'].mean():.2f}")
    print(f"  Median: {df['performance_clean'].median():.2f}")

# Before/after cleaning examples
print(f"\nBefore/after cleaning value examples:")
sample = df[df['performance_final'].notna()].head(5)
for idx, row in sample.iterrows():
    print(f"  {row['model_id']:50} : {str(row['performance_final'])[:30]:30} → {row['performance_clean']:.2f}")

# Save cleaned dataset
df.to_csv("merged_analysis_dataset_clean.csv", index=False)
print(f"\nCleaned dataset saved: merged_analysis_dataset_clean.csv")

# Analysis by explainability category
print(f"\n" + "="*80)
print("ANALYSIS BY EXPLAINABILITY CATEGORY")
print("="*80)

for cat in ['lightweight', 'medium', 'complex']:
    subset = df[df['explicability_final'] == cat]
    perf_values = subset['performance_clean'].dropna()
    if len(perf_values) > 0:
        print(f"  {cat:15} : M = {perf_values.mean():.2f}, SD = {perf_values.std():.2f}, n = {len(perf_values)}")
    else:
        print(f"  {cat:15} : No performance data")

print(f"\nTotal models with complete data (performance + explainability):")
complete = df.dropna(subset=['performance_clean', 'explicability_ordinal'])
print(f"  {len(complete)} models")