# figure_supp_timeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Charger les données
df = pd.read_csv("comprehensive_nlp_models_database.csv")

# Créer la figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Couleurs et styles
arch_colors = {'encoder': '#1f77b4', 'encoder-decoder': '#2ca02c', 'decoder': '#d62728'}
size_scale = 100  # Échelle pour la taille des points

# 1. Timeline principale : Performance vs Année
for arch in df['architecture'].unique():
    subset = df[df['architecture'] == arch]
    
    # Taille des points basée sur le log des paramètres
    sizes = np.log10(subset['parameters_M'] + 1) * size_scale
    
    axes[0,0].scatter(subset['year'], subset['score'],
                     s=sizes, color=arch_colors[arch],
                     alpha=0.7, edgecolors='black', linewidth=0.5,
                     label=f'{arch} (n={len(subset)})')
    
    # Ajouter des étiquettes pour les modèles importants
    for idx, row in subset.iterrows():
        if row['parameters_M'] > 10000 or row['score'] > 88:  # Grands ou performants
            axes[0,0].annotate(row['model_name'].split('-')[0],  # Nom court
                             (row['year'], row['score']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8, alpha=0.8)

axes[0,0].set_xlabel('Year')
axes[0,0].set_ylabel('Performance')
axes[0,0].set_title('A) Performance Timeline by Architecture and Size')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Évolution de l'explicabilité dans le temps
for arch in df['architecture'].unique():
    subset = df[df['architecture'] == arch]
    axes[0,1].scatter(subset['year'], subset['explainability_score'],
                     s=100, color=arch_colors[arch],
                     alpha=0.7, edgecolors='black', linewidth=0.5,
                     label=f'{arch}')
    
    # Ligne de tendance
    if len(subset) > 2:
        z = np.polyfit(subset['year'], subset['explainability_score'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['year'].min(), subset['year'].max(), 100)
        axes[0,1].plot(x_range, p(x_range), color=arch_colors[arch],
                      linestyle='--', alpha=0.5, linewidth=1)

axes[0,1].set_xlabel('Year')
axes[0,1].set_ylabel('Explainability Score (1-5)')
axes[0,1].set_title('B) Explainability Evolution Over Time')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Timeline des innovations (graphique à barres)
# Grouper par année et architecture
yearly_counts = df.groupby(['year', 'architecture']).size().unstack(fill_value=0)

# Créer un stacked bar chart
bottom_values = np.zeros(len(yearly_counts))
for i, arch in enumerate(['encoder', 'encoder-decoder', 'decoder']):
    if arch in yearly_counts.columns:
        axes[1,0].bar(yearly_counts.index, yearly_counts[arch], 
                     bottom=bottom_values, color=arch_colors[arch],
                     label=arch, alpha=0.8)
        bottom_values += yearly_counts[arch].values

axes[1,0].set_xlabel('Year')
axes[1,0].set_ylabel('Number of Models')
axes[1,0].set_title('C) Model Releases by Year and Architecture')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3, axis='y')

# 4. Heatmap des performances par année et architecture
# Créer une matrice de performances moyennes
heatmap_data = pd.pivot_table(df, values='score', 
                              index='architecture', 
                              columns='year', 
                              aggfunc='mean')

# Réorganiser les colonnes et lignes
heatmap_data = heatmap_data.reindex(index=['encoder', 'encoder-decoder', 'decoder'])
heatmap_data = heatmap_data.sort_index(axis=1)  # Trier les colonnes

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes[1,1], cbar_kws={'label': 'Average Performance'})
axes[1,1].set_title('D) Average Performance by Year and Architecture')
axes[1,1].set_xlabel('Year')
axes[1,1].set_ylabel('Architecture')

plt.suptitle('Temporal Analysis of NLP Model Development (2019-2023)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Sauvegarder
plt.savefig('figure_supp_timeline_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figure supplémentaire sauvegardée: figure_supp_timeline_analysis.png")

# Générer aussi une table chronologique
print("\n📊 Chronologie des modèles par année:")
for year in sorted(df['year'].unique()):
    year_models = df[df['year'] == year]
    print(f"\n{year}:")
    for _, row in year_models.iterrows():
        print(f"  • {row['model_name']}: Perf={row['score']:.1f}, Expl={row['explainability_score']:.1f}, Params={row['parameters_M']:,.0f}M")