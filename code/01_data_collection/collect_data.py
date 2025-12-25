from huggingface_hub import list_models, model_info, HfApi
import pandas as pd
import time
import re
import json

print("Recherche de modèles de classification de texte populaires...")
models = list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=100
)

model_ids = [model.id for model in models]
print(f"Nombre de modèles à traiter : {len(model_ids)}")

# Catégories pour notre proxy d'explicabilité basé sur l'architecture
model_type_categories = {
    'distil': 'lightweight',       # Modèles distillés, plus simples
    'tiny': 'lightweight',
    'mini': 'lightweight',
    'small': 'lightweight',
    'base': 'medium',
    'large': 'complex',
    'roberta': 'complex',
    'deberta': 'complex',
    'electra': 'medium',
    'albert': 'lightweight',       # Albert partage ses paramètres
    'xlm': 'complex',
    'bert': 'medium'               # Par défaut pour BERT
}

# Initialiser l'API pour récupérer le README
api = HfApi()

data = []

for i, mid in enumerate(model_ids):
    try:
        print(f"[{i+1}/{len(model_ids)}] Traitement de : {mid}")
        info = model_info(mid)
        model_data = {
            "model_id": mid,
            "pipeline_tag": info.pipeline_tag or "N/A",
            "downloads": info.downloads or 0,
            "likes": info.likes or 0,
        }

        # --- 1. EXTRACTION DES PERFORMANCES (Correction majeure) ---
        card_data = info.cardData
        perf_extracted = False
        
        if card_data:
            try:
                # Convertir cardData en dictionnaire si possible
                if hasattr(card_data, 'to_dict'):
                    card_dict = card_data.to_dict()
                else:
                    card_dict = dict(card_data) if hasattr(card_data, '__dict__') else {}
                
                # Essayer d'extraire les performances via model-index
                if isinstance(card_dict, dict) and card_dict:
                    # Rechercher récursivement des données de performance
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
                    
                    # Recherche spécifique de model-index (structure standard)
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
                print(f"   ⚠️  Erreur lors de l'extraction des performances pour {mid}: {str(e)[:50]}")

        # --- 2. PROXY POUR L'EXPLICABILITÉ : TYPE D'ARCHITECTURE ---
        model_name_lower = mid.lower()
        model_type = "unknown"
        for key, category in model_type_categories.items():
            if key in model_name_lower:
                model_type = category
                break
        model_data["model_type_proxy"] = model_type

        # --- 3. EXTRACTION DE LA TAILLE (Paramètres) ---
        # Essayer de récupérer le fichier de configuration
        try:
            config_url = f"https://huggingface.co/{mid}/raw/main/config.json"
            import requests
            response = requests.get(config_url, timeout=5)
            if response.status_code == 200:
                config = response.json()
                if 'num_parameters' in config:
                    model_data["num_parameters"] = config['num_parameters']
                elif 'hidden_size' in config and 'num_hidden_layers' in config:
                    # Estimation grossière pour BERT-like
                    hidden_size = config['hidden_size']
                    num_layers = config['num_hidden_layers']
                    vocab_size = config.get('vocab_size', 30522)
                    model_data["estimated_parameters"] = (hidden_size * hidden_size * 12 * num_layers) + (hidden_size * vocab_size)
        except:
            model_data["num_parameters"] = "N/A"

        # --- 4. PRÉSENCE D'UNE FICHE DÉTAILLÉE ---
        # Récupérer le README pour vérifier la présence de sections importantes
        try:
            readme_content = api.model_info(mid).cardData
            if readme_content:
                readme_text = str(readme_content).lower()
                has_sections = any(section in readme_text for section in 
                                  ["model card", "model details", "intended use", "limitations", 
                                   "training data", "evaluation", "results"])
                model_data["has_model_card_signal"] = int(has_sections)
                
                # Recherche de valeurs numériques de performance dans le README
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
                            # Ne pas ajouter si déjà extrait via model-index
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
        
        # Pause pour respecter les limites de l'API
        time.sleep(0.5)

    except Exception as e:
        print(f"   ⚠️  Erreur sur {mid} : {str(e)[:100]}...")
        continue

# Création du DataFrame
df = pd.DataFrame(data)
output_file = "huggingface_models_with_proxy.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Données sauvegardées dans '{output_file}'.")

# Analyse
print("\n" + "="*60)
print("ANALYSE DES DONNÉES COLLECTÉES :")
print("="*60)
print(f"1. Total des modèles : {len(df)}")

# Compter les modèles avec au moins une métrique de performance
perf_cols = [col for col in df.columns if col.startswith('perf_')]
print(f"2. Colonnes de performance extraites : {len(perf_cols)}")

if perf_cols:
    df_with_perf = df.dropna(subset=perf_cols, how='all')
    print(f"3. Modèles avec au moins une métrique de performance : {len(df_with_perf)}")
    
    if len(df_with_perf) > 0:
        # Afficher un exemple de métrique
        sample_model = df_with_perf.iloc[0]
        sample_perf = {col: sample_model[col] for col in perf_cols if pd.notna(sample_model[col])}
        print(f"   Exemple (premier modèle) : {list(sample_perf.keys())[:3]}")

# Distribution du proxy de type
print(f"4. Distribution du proxy 'model_type_proxy' :")
print(df['model_type_proxy'].value_counts())

# Distribution des signaux de fiche modèle
print(f"5. Modèles avec fiche technique détaillée : {df['has_model_card_signal'].sum()}/{len(df)}")

print("="*60)
print("POUR VOTRE ANALYSE :")
print("1. Ouvrez le fichier CSV généré.")
print("2. Utilisez les colonnes 'perf_*' pour la variable de Performance (P).")
print("3. Utilisez la colonne 'model_type_proxy' (codée ordinalement : lightweight < medium < complex) comme premier proxy pour l'Explicabilité (E).")
print("4. Vous pouvez croiser cela avec 'has_model_card_signal'.")
print("="*60)

# Analyse supplémentaire
print("\n" + "="*70)
print("INVESTIGATION DÉTAILLÉE :")
print("="*70)

# Vérifier quelques modèles spécifiques
test_models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
print("\nVérification de modèles connus :")
for mid in test_models:
    if mid in df['model_id'].values:
        row = df[df['model_id'] == mid].iloc[0]
        perf_cols_model = [col for col in perf_cols if col in row and pd.notna(row[col])]
        print(f"  {mid}: {len(perf_cols_model)} métriques de performance")
        if perf_cols_model:
            for col in perf_cols_model[:2]:  # Afficher 2 métriques max
                print(f"    - {col}: {row[col]}")

print("\n" + "="*70)
print("PROCHAINES ÉTAPES POUR L'ARTICLE :")
print("="*70)
print("1. Fusionner ces données avec votre base manuelle")
print("2. Nettoyer et normaliser les valeurs de performance")
print("3. Coder les variables pour l'analyse statistique")
print("4. Exécuter les analyses de corrélation et régression")
print("5. Créer les visualisations pour l'article")