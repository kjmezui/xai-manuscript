# figure_supp_architecture_correlation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot avec régression par architecture
architectures = df['architecture'].unique()
colors = {'encoder': 'blue', 'encoder-decoder': 'green', 'decoder': 'red'}
markers = {'encoder': 'o', 'encoder-decoder': 's', 'decoder': '^'}

# Plot principal
for arch in architectures:
    subset = df[df['architecture'] == arch]
    axes[0,0].scatter(subset['explainability_score'], subset['score'],
                     color=colors[arch], marker=markers[arch], s=100,
                     label=f'{arch} (n={len(subset)})', alpha=0.8, edgecolors='black')
    
    # Ligne de régression par architecture
    if len(subset) > 2:
        z = np.polyfit(subset['explainability_score'], subset['score'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['explainability_score'].min(), 
                              subset['explainability_score'].max(), 100)
        axes[0,0].plot(x_range, p(x_range), color=colors[arch], 
                      linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Calculer la corrélation
        corr, p_val = stats.spearmanr(subset['explainability_score'], subset['score'])
        print(f"{arch}: ρ = {corr:.3f}, p = {p_val:.4f}")

axes[0,0].set_xlabel('Explainability Score (1-5)')
axes[0,0].set_ylabel('Performance')
axes[0,0].set_title('A) Performance vs Explainability by Architecture')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Box plot comparatif
df_box = df.copy()
df_box['Performance'] = df['score']
df_box['Architecture'] = df['architecture']

sns.boxplot(data=df_box, x='Architecture', y='Performance', 
            order=['encoder', 'encoder-decoder', 'decoder'],
            palette=colors, ax=axes[0,1])
sns.stripplot(data=df_box, x='Architecture', y='Performance', 
              order=['encoder', 'encoder-decoder', 'decoder'],
              color='black', alpha=0.6, size=6, jitter=0.2, ax=axes[0,1])
axes[0,1].set_title('B) Performance Distribution by Architecture')
axes[0,1].set_ylabel('Performance')
axes[0,1].grid(True, alpha=0.3, axis='y')

# 3. Violin plot avec répartition
sns.violinplot(data=df_box, x='Architecture', y='Performance',
               order=['encoder', 'encoder-decoder', 'decoder'],
               palette=colors, inner='quartile', ax=axes[1,0])
axes[1,0].set_title('C) Performance Density by Architecture')
axes[1,0].set_ylabel('Performance')
axes[1,0].grid(True, alpha=0.3, axis='y')

# 4. Matrice de corrélation par architecture
# Créer une figure de corrélation pour chaque architecture
corr_data = df[['score', 'explainability_score', 'complexity_score', 'parameters_M']]

# Normaliser les paramètres pour l'échelle
corr_data['log_parameters'] = np.log10(corr_data['parameters_M'] + 1)

# Calculer les corrélations
corr_matrix = corr_data[['score', 'explainability_score', 'complexity_score', 'log_parameters']].corr(method='spearman')

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, ax=axes[1,1], 
            cbar_kws={'label': "Spearman's ρ"})
axes[1,1].set_title('D) Correlation Matrix (All Models)')
axes[1,1].set_xticklabels(['Performance', 'Explainability', 'Complexity', 'log(Params)'], rotation=45)
axes[1,1].set_yticklabels(['Performance', 'Explainability', 'Complexity', 'log(Params)'], rotation=0)

plt.suptitle('Detailed Analysis by Architectural Family', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Sauvegarder
plt.savefig('figure_supp_architecture_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figure supplémentaire sauvegardée: figure_supp_architecture_analysis.png")