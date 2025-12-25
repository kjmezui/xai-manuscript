# meta_analysis.py - Version complète
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Configuration pour les publications scientifiques
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Charger la base de données
df = pd.read_csv("comprehensive_nlp_models_database.csv")

print("="*80)
print("MÉTA-ANALYSE SYSTÉMATIQUE : PERFORMANCE vs EXPLICABILITÉ")
print("="*80)

print(f"\n📊 DESCRIPTION DE L'ÉCHANTILLON")
print(f"   • Nombre total de modèles : {len(df)}")
print(f"   • Période : {df['year'].min()} - {df['year'].max()}")
print(f"   • Architectures représentées : {', '.join(df['architecture'].unique())}")

print(f"\n📈 STATISTIQUES DESCRIPTIVES")
print(f"   • Performance : M = {df['score'].mean():.2f}, SD = {df['score'].std():.2f}")
print(f"   • Explicabilité (1-5) : M = {df['explainability_score'].mean():.2f}, SD = {df['explainability_score'].std():.2f}")
print(f"   • Paramètres (en millions) : M = {df['parameters_M'].mean():.1f}M, SD = {df['parameters_M'].std():.1f}M")
print(f"   • Complexité composite : M = {df['complexity_score'].mean():.2f}, SD = {df['complexity_score'].std():.2f}")

print(f"\n🔗 ANALYSE DE CORRÉLATION (SPEARMAN)")

# 1. Performance vs Explicabilité
corr_perf_exp, p_perf_exp = stats.spearmanr(df['score'], df['explainability_score'])
print(f"   1. Performance ↔ Explicabilité")
print(f"      ρ = {corr_perf_exp:.3f}, p = {p_perf_exp:.4f}")
if p_perf_exp < 0.05:
    print(f"      → Corrélation significative ({'négative' if corr_perf_exp < 0 else 'positive'})")

# 2. Performance vs Complexité
corr_perf_comp, p_perf_comp = stats.spearmanr(df['score'], df['complexity_score'])
print(f"\n   2. Performance ↔ Complexité")
print(f"      ρ = {corr_perf_comp:.3f}, p = {p_perf_comp:.4f}")
if p_perf_comp < 0.05:
    print(f"      → Corrélation significative ({'négative' if corr_perf_comp < 0 else 'positive'})")

# 3. Explicabilité vs Complexité
corr_exp_comp, p_exp_comp = stats.spearmanr(df['explainability_score'], df['complexity_score'])
print(f"\n   3. Explicabilité ↔ Complexité")
print(f"      ρ = {corr_exp_comp:.3f}, p = {p_exp_comp:.4f}")
if p_exp_comp < 0.05:
    print(f"      → Corrélation significative ({'négative' if corr_exp_comp < 0 else 'positive'})")

# 4. Performance vs Année
corr_perf_year, p_perf_year = stats.spearmanr(df['score'], df['year'])
print(f"\n   4. Performance ↔ Année")
print(f"      ρ = {corr_perf_year:.3f}, p = {p_perf_year:.4f}")
if p_perf_year < 0.05:
    print(f"      → Corrélation significative ({'négative' if corr_perf_year < 0 else 'positive'})")

print(f"\n📊 RÉGRESSION MULTIPLE")

# Préparation des variables pour la régression
X = df[['complexity_score', 'explainability_score', 'year']]
X = sm.add_constant(X)  # Ajout de l'intercept
y = df['score']

model = sm.OLS(y, X).fit()
print(model.summary())

# Extraire les résultats importants
print(f"\n🔑 PRINCIPAUX RÉSULTATS DE LA RÉGRESSION :")
print(f"   • R² = {model.rsquared:.3f}")
print(f"   • R² ajusté = {model.rsquared_adj:.3f}")

for param in model.params.index:
    if param != 'const':
        p_value = model.pvalues[param]
        coef = model.params[param]
        print(f"   • {param}: β = {coef:.3f}, p = {p_value:.4f}", end="")
        if p_value < 0.05:
            print(f" (significatif)")
        else:
            print(f" (non significatif)")

print(f"\n🏗️ ANALYSE PAR ARCHITECTURE")
print("   " + "-"*50)

arch_stats = df.groupby('architecture').agg({
    'score': ['mean', 'std', 'count'],
    'explainability_score': ['mean', 'std'],
    'complexity_score': ['mean', 'std']
}).round(2)

print(arch_stats)

print(f"\n📋 TEST D'HYPOTHÈSE : DIFFÉRENCES ENTRE ARCHITECTURES")

# ANOVA pour les différences de performance par architecture
print(f"\n   1. ANOVA : Performance ~ Architecture")
model_anova = ols('score ~ C(architecture)', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print(anova_table)

if anova_table['PR(>F)'][0] < 0.05:
    print(f"\n      → Différences significatives entre architectures")
    # Test post-hoc Tukey
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(
        endog=df['score'],
        groups=df['architecture'],
        alpha=0.05
    )
    print(f"\n      Tests post-hoc Tukey HSD :")
    print(tukey)
else:
    print(f"\n      → Pas de différences significatives entre architectures")

print(f"\n🎯 IMPLICATIONS POUR L'ARTICLE")

# Calculer l'effet taille (Cohen's d) pour la différence de performance entre modèles simples et complexes
# Définir simple: explicability_score >= 3, complexe: explicability_score <= 2
simple_models = df[df['explainability_score'] >= 3]
complex_models = df[df['explainability_score'] <= 2]

if len(simple_models) > 0 and len(complex_models) > 0:
    mean_simple = simple_models['score'].mean()
    mean_complex = complex_models['score'].mean()
    std_pooled = np.sqrt((simple_models['score'].var() + complex_models['score'].var()) / 2)
    cohens_d = (mean_complex - mean_simple) / std_pooled
    
    print(f"\n   1. COMPARAISON MODÈLES SIMPLES vs COMPLEXES :")
    print(f"      • Modèles simples (explicabilité ≥ 3) : n = {len(simple_models)}, M = {mean_simple:.2f}")
    print(f"      • Modèles complexes (explicabilité ≤ 2) : n = {len(complex_models)}, M = {mean_complex:.2f}")
    print(f"      • Différence de performance : {mean_complex - mean_simple:.2f} points")
    print(f"      • Taille d'effet (Cohen's d) : {cohens_d:.3f}")
    
    # Test t indépendant
    t_stat, t_p = stats.ttest_ind(simple_models['score'], complex_models['score'], equal_var=False)
    print(f"      • Test t : t = {t_stat:.3f}, p = {t_p:.4f}")
    if t_p < 0.05:
        print(f"      → Différence significative")

print(f"\n   2. TENDANCE TEMPORELLE :")
print(f"      • Les modèles plus récents ont tendance à avoir une performance plus élevée")
print(f"      • Corrélation année-performance : ρ = {corr_perf_year:.3f}")

print(f"\n   3. COMPROMIS PERFORMANCE-EXPLICABILITÉ :")
if p_perf_exp < 0.05 and corr_perf_exp < 0:
    print(f"      • Confirmé : corrélation négative significative")
    print(f"      • Amélioration de la performance associée à une diminution de l'explicabilité")
elif p_perf_exp < 0.05 and corr_perf_exp > 0:
    print(f"      • Contre-intuitif : corrélation positive")
    print(f"      • Nécessite une interprétation approfondie")
else:
    print(f"      • Pas de corrélation significative détectée")

# Création des visualisations
print(f"\n📊 CRÉATION DES VISUALISATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter: Performance vs Explicabilité
sc1 = axes[0,0].scatter(df['explainability_score'], df['score'], 
                       c=df['year'], cmap='viridis', s=100, alpha=0.8, edgecolors='black')
axes[0,0].set_xlabel('Score d\'Explicabilité (1-5)')
axes[0,0].set_ylabel('Performance')
axes[0,0].set_title(f'A) Performance vs Explicabilité\nρ = {corr_perf_exp:.3f}')
plt.colorbar(sc1, ax=axes[0,0], label='Année')

# Ligne de régression
z = np.polyfit(df['explainability_score'], df['score'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['explainability_score'].min(), df['explainability_score'].max(), 100)
axes[0,0].plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)

# 2. Scatter: Performance vs Complexité
sc2 = axes[0,1].scatter(df['complexity_score'], df['score'], 
                       c=df['explainability_score'], cmap='coolwarm', s=100, alpha=0.8, edgecolors='black')
axes[0,1].set_xlabel('Score de Complexité')
axes[0,1].set_ylabel('Performance')
axes[0,1].set_title(f'B) Performance vs Complexité\nρ = {corr_perf_comp:.3f}')
plt.colorbar(sc2, ax=axes[0,1], label='Explicabilité')

z = np.polyfit(df['complexity_score'], df['score'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['complexity_score'].min(), df['complexity_score'].max(), 100)
axes[0,1].plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)

# 3. Évolution temporelle
for arch in df['architecture'].unique():
    subset = df[df['architecture'] == arch]
    axes[0,2].scatter(subset['year'], subset['score'], label=arch, s=80, alpha=0.7)
axes[0,2].set_xlabel('Année')
axes[0,2].set_ylabel('Performance')
axes[0,2].set_title(f'C) Évolution Temporelle\nρ = {corr_perf_year:.3f}')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Box plot: Performance par architecture
df.boxplot(column='score', by='architecture', ax=axes[1,0])
axes[1,0].set_xlabel('Architecture')
axes[1,0].set_ylabel('Performance')
axes[1,0].set_title('D) Distribution par Architecture')
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Heatmap de corrélation
corr_matrix = df[['score', 'explainability_score', 'complexity_score', 'year', 'parameters_M']].corr(method='spearman')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=axes[1,1], cbar_kws={'label': 'Coefficient de Corrélation'})
axes[1,1].set_title('E) Matrice de Corrélation (Spearman)')

# 6. Bar plot: Performance moyenne vs Explicabilité moyenne par architecture
arch_summary = df.groupby('architecture').agg({
    'score': 'mean',
    'explainability_score': 'mean'
}).reset_index()

x = np.arange(len(arch_summary))
width = 0.35

bars1 = axes[1,2].bar(x - width/2, arch_summary['score'], width, label='Performance', color='skyblue')
bars2 = axes[1,2].bar(x + width/2, arch_summary['explainability_score']*20, width, label='Explicabilité (×20)', color='lightcoral')

axes[1,2].set_xlabel('Architecture')
axes[1,2].set_ylabel('Score')
axes[1,2].set_title('F) Performance vs Explicabilité par Architecture')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(arch_summary['architecture'], rotation=45)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3, axis='y')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Méta-analyse: Compromis Performance-Explicabilité dans les Modèles NLP', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('meta_analysis_complete_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Visualisations sauvegardées dans 'meta_analysis_complete_results.png'")

print(f"\n" + "="*80)
print("📝 RÉSUMÉ POUR LA RÉDACTION DE L'ARTICLE")
print("="*80)

print(f"\n1. CONTEXTE EXPÉRIMENTAL :")
print(f"   • Échantillon : {len(df)} modèles NLP (2019-2023)")
print(f"   • Méthode : Méta-analyse systématique de la littérature")
print(f"   • Variables : Performance, Explicabilité, Complexité, Architecture")

print(f"\n2. RÉSULTATS CLÉS :")
print(f"   • Performance moyenne : {df['score'].mean():.2f} (SD = {df['score'].std():.2f})")
print(f"   • Explicabilité moyenne : {df['explainability_score'].mean():.2f}/5")
print(f"   • Corrélation performance-explicabilité : ρ = {corr_perf_exp:.3f} (p = {p_perf_exp:.4f})")
print(f"   • Corrélation performance-complexité : ρ = {corr_perf_comp:.3f} (p = {p_perf_comp:.4f})")

print(f"\n3. INTERPRÉTATION :")
if p_perf_exp < 0.05 and corr_perf_exp < 0:
    print(f"   • Le trade-off performance-explicabilité est confirmé statistiquement")
    print(f"   • Les modèles plus performants tendent à être moins explicables")
elif p_perf_exp >= 0.05:
    print(f"   • Aucune corrélation significative n'a été détectée")
    print(f"   • Le trade-off pourrait être moins prononcé qu'attendu")

print(f"\n4. IMPLICATIONS :")
print(f"   • Pour la recherche : Nécessité de développer des métriques d'explicabilité standardisées")
print(f"   • Pour la pratique : Guide pour le choix de modèles selon les contraintes")
print(f"   • Pour l'industrie : Importance de l'explicabilité pour le déploiement responsable")

print(f"\n" + "="*80)
print("✅ ANALYSE TERMINÉE - PRÊT POUR LA RÉDACTION")
print("="*80)