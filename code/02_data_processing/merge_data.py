import pandas as pd
import numpy as np
import os

print("ÉTAPE 1 : Fusion des données...")

# Vérifier que les fichiers existent
required_files = ['manual_model_performance.csv', 'huggingface_models_with_proxy.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"❌ Fichiers manquants: {missing_files}")
    print("Veuillez d'abord exécuter:")
    print("1. manual_performance_dataset.py")
    print("2. collect_data.py (corrigé)")
    exit(1)

# Charger les données
print("Chargement des données...")
collected = pd.read_csv("huggingface_models_with_proxy.csv")
manual = pd.read_csv("manual_model_performance.csv")

print(f"Données collectées: {len(collected)} modèles")
print(f"Données manuelles: {len(manual)} modèles")

# Fusionner les données
print("Fusion des datasets...")
merged = pd.merge(
    collected, 
    manual[['model_id', 'performance_value', 'performance_metric', 'explicability_proxy', 'paper', 'notes']], 
    on='model_id', 
    how='left', 
    suffixes=('_collected', '_manual')
)

print(f"Dataset fusionné: {len(merged)} modèles")

# Créer une variable de performance unifiée
def get_best_performance(row):
    # Priorité 1: Données manuelles (plus fiables)
    if pd.notna(row.get('performance_value')):
        return row['performance_value']
    
    # Priorité 2: Données collectées
    perf_cols = [col for col in row.index if col.startswith('perf_')]
    for col in perf_cols:
        if pd.notna(row[col]):
            return row[col]
    
    return np.nan

merged['performance_final'] = merged.apply(get_best_performance, axis=1)

# Créer une variable d'explicabilité unifiée
def get_explicability_proxy(row):
    if pd.notna(row.get('explicability_proxy')):
        return row['explicability_proxy']
    elif pd.notna(row.get('model_type_proxy')):
        return row['model_type_proxy']
    else:
        return 'unknown'

merged['explicability_final'] = merged.apply(get_explicability_proxy, axis=1)

# Coder ordinalement l'explicabilité
explicability_mapping = {
    'lightweight': 1,
    'medium': 2, 
    'complex': 3,
    'unknown': np.nan
}
merged['explicability_ordinal'] = merged['explicability_final'].map(explicability_mapping)

# Sauvegarder
output_file = "merged_analysis_dataset.csv"
merged.to_csv(output_file, index=False)

print(f"\n✅ Dataset fusionné sauvegardé dans '{output_file}'")
print(f"   Modèles avec performance: {merged['performance_final'].notna().sum()}/{len(merged)}")
print(f"   Modèles avec explicabilité: {merged['explicability_ordinal'].notna().sum()}/{len(merged)}")

# Aperçu des données
print("\n📊 APERÇU DES DONNÉES FUSIONNÉES:")
print(merged[['model_id', 'performance_final', 'explicability_final', 'explicability_ordinal']].head(10))