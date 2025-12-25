# clean_data.py
import pandas as pd
import numpy as np
import re

print("="*80)
print("NETTOYAGE DES DONNÉES DE PERFORMANCE")
print("="*80)

# Charger les données
df = pd.read_csv("merged_analysis_dataset.csv")

print(f"Avant nettoyage : {len(df)} modèles")
print(f"Modèles avec performance : {df['performance_final'].notna().sum()}")

# Fonction pour nettoyer les valeurs de performance
def clean_performance_value(value):
    if pd.isna(value):
        return np.nan
    
    # Convertir en chaîne si ce n'est pas déjà le cas
    value_str = str(value).strip()
    
    # 1. Supprimer les pourcentages
    value_str = value_str.replace('%', '')
    
    # 2. Supprimer les espaces
    value_str = value_str.replace(' ', '')
    
    # 3. Gérer les cas comme "0.98855216855484120.982300884955752298.38"
    # Cette chaîne semble contenir plusieurs nombres collés
    # Essayons d'extraire le premier nombre valide
    matches = re.findall(r'\d+\.\d+', value_str)
    if matches:
        # Prendre le premier nombre trouvé
        try:
            return float(matches[0])
        except:
            pass
    
    # 4. Essayer de convertir directement en float
    try:
        return float(value_str)
    except:
        # 5. Si échec, essayer d'extraire n'importe quel nombre
        number_match = re.search(r'\d+\.?\d*', value_str)
        if number_match:
            try:
                return float(number_match.group())
            except:
                return np.nan
    
    return np.nan

# Nettoyer la colonne performance_final
df['performance_final_clean'] = df['performance_final'].apply(clean_performance_value)

# Vérifier aussi les autres colonnes de performance
perf_cols = [col for col in df.columns if col.startswith('perf_')]
for col in perf_cols:
    if col in df.columns:
        df[f'{col}_clean'] = df[col].apply(clean_performance_value)

# Calculer une performance finale améliorée
def get_best_performance_clean(row):
    # Priorité 1: Valeur nettoyée de performance_final
    if pd.notna(row.get('performance_final_clean')):
        return row['performance_final_clean']
    
    # Priorité 2: Valeurs nettoyées des autres colonnes de performance
    clean_perf_cols = [col for col in df.columns if col.endswith('_clean') and not col.startswith('performance_final')]
    for col in clean_perf_cols:
        if pd.notna(row[col]):
            return row[col]
    
    # Priorité 3: Essayons d'extraire des performances du texte des notes
    if pd.notna(row.get('notes')):
        notes_text = str(row['notes'])
        # Chercher des nombres dans les notes
        number_matches = re.findall(r'(\d+\.\d+)\s*%?', notes_text)
        if number_matches:
            try:
                # Prendre le plus grand nombre (probablement une performance)
                numbers = [float(match) for match in number_matches]
                if numbers:
                    # Filtrer les nombres plausibles (entre 0 et 100 pour les scores)
                    plausible = [n for n in numbers if 0 <= n <= 100]
                    if plausible:
                        return max(plausible)  # Prendre la meilleure performance
            except:
                pass
    
    return np.nan

df['performance_clean'] = df.apply(get_best_performance_clean, axis=1)

# Statistiques après nettoyage
print(f"\nAprès nettoyage :")
print(f"  Modèles avec performance nettoyée : {df['performance_clean'].notna().sum()}")
print(f"  Modèles avec explicabilité : {df['explicability_ordinal'].notna().sum()}")

# Distribution des performances nettoyées
print(f"\nDistribution des performances nettoyées :")
if df['performance_clean'].notna().sum() > 0:
    print(f"  Min : {df['performance_clean'].min():.2f}")
    print(f"  Max : {df['performance_clean'].max():.2f}")
    print(f"  Moyenne : {df['performance_clean'].mean():.2f}")
    print(f"  Médiane : {df['performance_clean'].median():.2f}")

# Exemples de valeurs nettoyées
print(f"\nExemples de valeurs avant/après nettoyage :")
sample = df[df['performance_final'].notna()].head(5)
for idx, row in sample.iterrows():
    print(f"  {row['model_id']:50} : {str(row['performance_final'])[:30]:30} → {row['performance_clean']:.2f}")

# Sauvegarder le dataset nettoyé
df.to_csv("merged_analysis_dataset_clean.csv", index=False)
print(f"\n✅ Dataset nettoyé sauvegardé : merged_analysis_dataset_clean.csv")

# Analyse par catégorie d'explicabilité
print(f"\n" + "="*80)
print("ANALYSE PAR CATÉGORIE D'EXPLICABILITÉ")
print("="*80)

for cat in ['lightweight', 'medium', 'complex']:
    subset = df[df['explicability_final'] == cat]
    perf_values = subset['performance_clean'].dropna()
    if len(perf_values) > 0:
        print(f"  {cat:15} : M = {perf_values.mean():.2f}, SD = {perf_values.std():.2f}, n = {len(perf_values)}")
    else:
        print(f"  {cat:15} : Aucune donnée de performance")

print(f"\nTotal des modèles avec données complètes (performance + explicabilité) :")
complete = df.dropna(subset=['performance_clean', 'explicability_ordinal'])
print(f"  {len(complete)} modèles")