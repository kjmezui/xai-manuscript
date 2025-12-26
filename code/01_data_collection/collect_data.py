# collect_data.py
from huggingface_hub import list_models, model_info, HfApi
import pandas as pd
import time
import re
import json

print("Searching for popular text classification models...")
models = list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=100
)

model_ids = [model.id for model in models]
print(f"Number of models to process: {len(model_ids)}")

# Categories for our explainability proxy based on architecture
model_type_categories = {
    'distil': 'lightweight',       # Distilled models, simpler
    'tiny': 'lightweight',
    'mini': 'lightweight',
    'small': 'lightweight',
    'base': 'medium',
    'large': 'complex',
    'roberta': 'complex',
    'deberta': 'complex',
    'electra': 'medium',
    'albert': 'lightweight',       # Albert shares parameters
    'xlm': 'complex',
    'bert': 'medium'               # Default for BERT
}

# Initialize API to fetch README
api = HfApi()

data = []

for i, mid in enumerate(model_ids):
    try:
        print(f"[{i+1}/{len(model_ids)}] Processing: {mid}")
        info = model_info(mid)
        model_data = {
            "model_id": mid,
            "pipeline_tag": info.pipeline_tag or "N/A",
            "downloads": info.downloads or 0,
            "likes": info.likes or 0,
        }

        # --- 1. PERFORMANCE EXTRACTION (Major correction) ---
        card_data = info.cardData
        perf_extracted = False
        
        if card_data:
            try:
                # Convert cardData to dictionary if possible
                if hasattr(card_data, 'to_dict'):
                    card_dict = card_data.to_dict()
                else:
                    card_dict = dict(card_data) if hasattr(card_data, '__dict__') else {}
                
                # Try to extract performance via model-index
                if isinstance(card_dict, dict) and card_dict:
                    # Recursively search for performance data
                    def extract_performance(data, prefix=""):
                        performances = {}
                        if isinstance(data, dict):
                            for key, value in data.items():
                                new_prefix = f"{prefix}_{key}" if prefix else key
                                if isinstance(value, (int, float)) and any(metric in key.lower() for metric in ['accuracy', 'f1', 'score', 'precision', 'recall', 'auc', 'glue', 'squad', 'bleu', 'rouge']):
                                    col_name = f"perf_{new_prefix}".replace(" ", "_").replace("-", "_").lower()
                                    performances[col_name] = value
                                elif isinstance(value, (dict, list)):
                                    performances.update(extract_performance(value, new_prefix))
                        elif isinstance(data, list):
                            for idx, item in enumerate(data):
                                performances.update(extract_performance(item, f"{prefix}_{idx}"))
                        return performances
                    
                    perf_data = extract_performance(card_dict)
                    for key, value in perf_data.items():
                        model_data[key] = value
                        perf_extracted = True
                    
                    # Specific search for model-index (standard structure)
                    if 'model-index' in card_dict:
                        model_index = card_dict['model-index']
                        if isinstance(model_index, list):
                            for item in model_index:
                                if isinstance(item, dict) and 'results' in item:
                                    results = item['results']
                                    if isinstance(results, list):
                                        for result in results:
                                            if isinstance(result, dict) and 'metrics' in result:
                                                metrics = result['metrics']
                                                if isinstance(metrics, list):
                                                    for metric in metrics:
                                                        if isinstance(metric, dict) and 'type' in metric and 'value' in metric:
                                                            metric_name = metric['type'].lower().replace(" ", "_")
                                                            value = metric['value']
                                                            col_name = f"perf_{metric_name}"
                                                            model_data[col_name] = value
                                                            perf_extracted = True
            except Exception as e:
                print(f"Error extracting performance for {mid}: {str(e)[:50]}")

        # --- 2. EXPLAINABILITY PROXY: ARCHITECTURE TYPE ---
        model_name_lower = mid.lower()
        model_type = "unknown"
        for key, category in model_type_categories.items():
            if key in model_name_lower:
                model_type = category
                break
        model_data["model_type_proxy"] = model_type

        # --- 3. SIZE EXTRACTION (Parameters) ---
        # Try to fetch configuration file
        try:
            config_url = f"https://huggingface.co/{mid}/raw/main/config.json"
            import requests
            response = requests.get(config_url, timeout=5)
            if response.status_code == 200:
                config = response.json()
                if 'num_parameters' in config:
                    model_data["num_parameters"] = config['num_parameters']
                elif 'hidden_size' in config and 'num_hidden_layers' in config:
                    # Rough estimation for BERT-like models
                    hidden_size = config['hidden_size']
                    num_layers = config['num_hidden_layers']
                    vocab_size = config.get('vocab_size', 30522)
                    model_data["estimated_parameters"] = (hidden_size * hidden_size * 12 * num_layers) + (hidden_size * vocab_size)
        except:
            model_data["num_parameters"] = "N/A"

        # --- 4. PRESENCE OF DETAILED DOCUMENTATION ---
        # Fetch README to check for important sections
        try:
            readme_content = api.model_info(mid).cardData
            if readme_content:
                readme_text = str(readme_content).lower()
                has_sections = any(section in readme_text for section in 
                                  ["model card", "model details", "intended use", "limitations", 
                                   "training data", "evaluation", "results"])
                model_data["has_model_card_signal"] = int(has_sections)
                
                # Search for numerical performance values in README
                performance_patterns = [
                    r'accuracy[\s:]*([0-9]*\.?[0-9]+)%?',
                    r'f1[\s:]*([0-9]*\.?[0-9]+)%?',
                    r'score[\s:]*([0-9]*\.?[0-9]+)%?',
                    r'([0-9]*\.?[0-9]+)%\s*(accuracy|f1|score)'
                ]
                
                for pattern in performance_patterns:
                    matches = re.findall(pattern, readme_text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            value = match[0]
                        else:
                            value = match
                        
                        try:
                            perf_value = float(value)
                            # Don't add if already extracted via model-index
                            if not perf_extracted:
                                model_data[f"perf_text_extracted"] = perf_value
                                perf_extracted = True
                        except:
                            pass
            else:
                model_data["has_model_card_signal"] = 0
        except:
            model_data["has_model_card_signal"] = 0

        data.append(model_data)
        
        # Pause to respect API limits
        time.sleep(0.5)

    except Exception as e:
        print(f"Error on {mid}: {str(e)[:100]}...")
        continue

# Create DataFrame
df = pd.DataFrame(data)
output_file = "huggingface_models_with_proxy.csv"
df.to_csv(output_file, index=False)
print(f"\nData saved to '{output_file}'.")

# Analysis
print("\n" + "="*60)
print("COLLECTED DATA ANALYSIS:")
print("="*60)
print(f"1. Total models: {len(df)}")

# Count models with at least one performance metric
perf_cols = [col for col in df.columns if col.startswith('perf_')]
print(f"2. Extracted performance columns: {len(perf_cols)}")

if perf_cols:
    df_with_perf = df.dropna(subset=perf_cols, how='all')
    print(f"3. Models with at least one performance metric: {len(df_with_perf)}")
    
    if len(df_with_perf) > 0:
        # Show a metric example
        sample_model = df_with_perf.iloc[0]
        sample_perf = {col: sample_model[col] for col in perf_cols if pd.notna(sample_model[col])}
        print(f"   Example (first model): {list(sample_perf.keys())[:3]}")

# Proxy type distribution
print(f"4. 'model_type_proxy' distribution:")
print(df['model_type_proxy'].value_counts())

# Model card signal distribution
print(f"5. Models with detailed technical sheet: {df['has_model_card_signal'].sum()}/{len(df)}")

print("="*60)
print("FOR YOUR ANALYSIS:")
print("1. Open the generated CSV file.")
print("2. Use the 'perf_*' columns for the Performance variable (P).")
print("3. Use the 'model_type_proxy' column (ordinal coding: lightweight < medium < complex) as first proxy for Explainability (E).")
print("4. You can cross-reference this with 'has_model_card_signal'.")
print("="*60)

# Additional analysis
print("\n" + "="*70)
print("DETAILED INVESTIGATION:")
print("="*70)

# Check specific models
test_models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
print("\nChecking known models:")
for mid in test_models:
    if mid in df['model_id'].values:
        row = df[df['model_id'] == mid].iloc[0]
        perf_cols_model = [col for col in perf_cols if col in row and pd.notna(row[col])]
        print(f"  {mid}: {len(perf_cols_model)} performance metrics")
        if perf_cols_model:
            for col in perf_cols_model[:2]:  # Show max 2 metrics
                print(f"    - {col}: {row[col]}")

print("\n" + "="*70)
print("NEXT STEPS FOR THE ARTICLE:")
print("="*70)
print("1. Merge this data with your manual database")
print("2. Clean and normalize performance values")
print("3. Code variables for statistical analysis")
print("4. Run correlation and regression analyses")
print("5. Create visualizations for the article")