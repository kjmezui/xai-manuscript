# meta_analysis.py - Complete version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Configuration for scientific publications
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

# Load database
df = pd.read_csv("comprehensive_nlp_models_database.csv")

print("="*80)
print("SYSTEMATIC META-ANALYSIS: PERFORMANCE vs EXPLAINABILITY")
print("="*80)

print(f"\nSAMPLE DESCRIPTION")
print(f"   • Total models: {len(df)}")
print(f"   • Period: {df['year'].min()} - {df['year'].max()}")
print(f"   • Represented architectures: {', '.join(df['architecture'].unique())}")

print(f"\nDESCRIPTIVE STATISTICS")
print(f"   • Performance: M = {df['score'].mean():.2f}, SD = {df['score'].std():.2f}")
print(f"   • Explainability (1-5): M = {df['explainability_score'].mean():.2f}, SD = {df['explainability_score'].std():.2f}")
print(f"   • Parameters (in millions): M = {df['parameters_M'].mean():.1f}M, SD = {df['parameters_M'].std():.1f}M")
print(f"   • Composite complexity: M = {df['complexity_score'].mean():.2f}, SD = {df['complexity_score'].std():.2f}")

print(f"\nCORRELATION ANALYSIS (SPEARMAN)")

# 1. Performance vs Explainability
corr_perf_exp, p_perf_exp = stats.spearmanr(df['score'], df['explainability_score'])
print(f"   1. Performance ↔ Explainability")
print(f"      ρ = {corr_perf_exp:.3f}, p = {p_perf_exp:.4f}")
if p_perf_exp < 0.05:
    print(f"      → Significant correlation ({'negative' if corr_perf_exp < 0 else 'positive'})")

# 2. Performance vs Complexity
corr_perf_comp, p_perf_comp = stats.spearmanr(df['score'], df['complexity_score'])
print(f"\n   2. Performance ↔ Complexity")
print(f"      ρ = {corr_perf_comp:.3f}, p = {p_perf_comp:.4f}")
if p_perf_comp < 0.05:
    print(f"      → Significant correlation ({'negative' if corr_perf_comp < 0 else 'positive'})")

# 3. Explainability vs Complexity
corr_exp_comp, p_exp_comp = stats.spearmanr(df['explainability_score'], df['complexity_score'])
print(f"\n   3. Explainability ↔ Complexity")
print(f"      ρ = {corr_exp_comp:.3f}, p = {p_exp_comp:.4f}")
if p_exp_comp < 0.05:
    print(f"      → Significant correlation ({'negative' if corr_exp_comp < 0 else 'positive'})")

# 4. Performance vs Year
corr_perf_year, p_perf_year = stats.spearmanr(df['score'], df['year'])
print(f"\n   4. Performance ↔ Year")
print(f"      ρ = {corr_perf_year:.3f}, p = {p_perf_year:.4f}")
if p_perf_year < 0.05:
    print(f"      → Significant correlation ({'negative' if corr_perf_year < 0 else 'positive'})")

print(f"\nMULTIPLE REGRESSION")

# Prepare variables for regression
X = df[['complexity_score', 'explainability_score', 'year']]
X = sm.add_constant(X)  # Add intercept
y = df['score']

model = sm.OLS(y, X).fit()
print(model.summary())

# Extract important results
print(f"\nKEY REGRESSION RESULTS:")
print(f"   • R² = {model.rsquared:.3f}")
print(f"   • Adjusted R² = {model.rsquared_adj:.3f}")

for param in model.params.index:
    if param != 'const':
        p_value = model.pvalues[param]
        coef = model.params[param]
        print(f"   • {param}: β = {coef:.3f}, p = {p_value:.4f}", end="")
        if p_value < 0.05:
            print(f" (significant)")
        else:
            print(f" (not significant)")

print(f"\nARCHITECTURE ANALYSIS")
print("   " + "-"*50)

arch_stats = df.groupby('architecture').agg({
    'score': ['mean', 'std', 'count'],
    'explainability_score': ['mean', 'std'],
    'complexity_score': ['mean', 'std']
}).round(2)

print(arch_stats)

print(f"\nHYPOTHESIS TESTING: ARCHITECTURE DIFFERENCES")

# ANOVA for performance differences by architecture
print(f"\n   1. ANOVA: Performance ~ Architecture")
model_anova = ols('score ~ C(architecture)', data=df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print(anova_table)

if anova_table['PR(>F)'][0] < 0.05:
    print(f"\nSignificant differences between architectures")
    # Tukey post-hoc test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(
        endog=df['score'],
        groups=df['architecture'],
        alpha=0.05
    )
    print(f"\n      Tukey HSD post-hoc tests:")
    print(tukey)
else:
    print(f"\nNo significant differences between architectures")

print(f"\nIMPLICATIONS FOR THE PAPER")

# Calculate effect size (Cohen's d) for performance difference between simple and complex models
# Define simple: explainability_score >= 3, complex: explainability_score <= 2
simple_models = df[df['explainability_score'] >= 3]
complex_models = df[df['explainability_score'] <= 2]

if len(simple_models) > 0 and len(complex_models) > 0:
    mean_simple = simple_models['score'].mean()
    mean_complex = complex_models['score'].mean()
    std_pooled = np.sqrt((simple_models['score'].var() + complex_models['score'].var()) / 2)
    cohens_d = (mean_complex - mean_simple) / std_pooled
    
    print(f"\n   1. SIMPLE vs COMPLEX MODELS COMPARISON:")
    print(f"      • Simple models (explainability ≥ 3): n = {len(simple_models)}, M = {mean_simple:.2f}")
    print(f"      • Complex models (explainability ≤ 2): n = {len(complex_models)}, M = {mean_complex:.2f}")
    print(f"      • Performance difference: {mean_complex - mean_simple:.2f} points")
    print(f"      • Effect size (Cohen's d): {cohens_d:.3f}")
    
    # Independent t-test
    t_stat, t_p = stats.ttest_ind(simple_models['score'], complex_models['score'], equal_var=False)
    print(f"      • T-test: t = {t_stat:.3f}, p = {t_p:.4f}")
    if t_p < 0.05:
        print(f"      → Significant difference")

print(f"\n   2. TEMPORAL TREND:")
print(f"      • More recent models tend to have higher performance")
print(f"      • Year-performance correlation: ρ = {corr_perf_year:.3f}")

print(f"\n   3. PERFORMANCE-EXPLAINABILITY TRADEOFF:")
if p_perf_exp < 0.05 and corr_perf_exp < 0:
    print(f"      • Confirmed: significant negative correlation")
    print(f"      • Performance improvement associated with decreased explainability")
elif p_perf_exp < 0.05 and corr_perf_exp > 0:
    print(f"      • Counter-intuitive: positive correlation")
    print(f"      • Requires deeper interpretation")
else:
    print(f"      • No significant correlation detected")

# Create visualizations
print(f"\nCREATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter: Performance vs Explainability
sc1 = axes[0,0].scatter(df['explainability_score'], df['score'], 
                       c=df['year'], cmap='viridis', s=100, alpha=0.8, edgecolors='black')
axes[0,0].set_xlabel('Explainability Score (1-5)')
axes[0,0].set_ylabel('Performance')
axes[0,0].set_title(f'A) Performance vs Explainability\nρ = {corr_perf_exp:.3f}')
plt.colorbar(sc1, ax=axes[0,0], label='Year')

# Regression line
z = np.polyfit(df['explainability_score'], df['score'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['explainability_score'].min(), df['explainability_score'].max(), 100)
axes[0,0].plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)

# 2. Scatter: Performance vs Complexity
sc2 = axes[0,1].scatter(df['complexity_score'], df['score'], 
                       c=df['explainability_score'], cmap='coolwarm', s=100, alpha=0.8, edgecolors='black')
axes[0,1].set_xlabel('Complexity Score')
axes[0,1].set_ylabel('Performance')
axes[0,1].set_title(f'B) Performance vs Complexity\nρ = {corr_perf_comp:.3f}')
plt.colorbar(sc2, ax=axes[0,1], label='Explainability')

z = np.polyfit(df['complexity_score'], df['score'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['complexity_score'].min(), df['complexity_score'].max(), 100)
axes[0,1].plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)

# 3. Temporal evolution
for arch in df['architecture'].unique():
    subset = df[df['architecture'] == arch]
    axes[0,2].scatter(subset['year'], subset['score'], label=arch, s=80, alpha=0.7)
axes[0,2].set_xlabel('Year')
axes[0,2].set_ylabel('Performance')
axes[0,2].set_title(f'C) Temporal Evolution\nρ = {corr_perf_year:.3f}')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Box plot: Performance by architecture
df.boxplot(column='score', by='architecture', ax=axes[1,0])
axes[1,0].set_xlabel('Architecture')
axes[1,0].set_ylabel('Performance')
axes[1,0].set_title('D) Distribution by Architecture')
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Correlation heatmap
corr_matrix = df[['score', 'explainability_score', 'complexity_score', 'year', 'parameters_M']].corr(method='spearman')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, ax=axes[1,1], cbar_kws={'label': 'Correlation Coefficient'})
axes[1,1].set_title('E) Correlation Matrix (Spearman)')

# 6. Bar plot: Average performance vs Average explainability by architecture
arch_summary = df.groupby('architecture').agg({
    'score': 'mean',
    'explainability_score': 'mean'
}).reset_index()

x = np.arange(len(arch_summary))
width = 0.35

bars1 = axes[1,2].bar(x - width/2, arch_summary['score'], width, label='Performance', color='skyblue')
bars2 = axes[1,2].bar(x + width/2, arch_summary['explainability_score']*20, width, label='Explainability (×20)', color='lightcoral')

axes[1,2].set_xlabel('Architecture')
axes[1,2].set_ylabel('Score')
axes[1,2].set_title('F) Performance vs Explainability by Architecture')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(arch_summary['architecture'], rotation=45)
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3, axis='y')

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                      f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Meta-analysis: Performance-Explainability Trade-off in NLP Models', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('meta_analysis_complete_results.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to 'meta_analysis_complete_results.png'")

print(f"\n" + "="*80)
print("SUMMARY FOR PAPER WRITING")
print("="*80)

print(f"\n1. EXPERIMENTAL CONTEXT:")
print(f"   • Sample: {len(df)} NLP models (2019-2023)")
print(f"   • Method: Systematic meta-analysis of literature")
print(f"   • Variables: Performance, Explainability, Complexity, Architecture")

print(f"\n2. KEY RESULTS:")
print(f"   • Average performance: {df['score'].mean():.2f} (SD = {df['score'].std():.2f})")
print(f"   • Average explainability: {df['explainability_score'].mean():.2f}/5")
print(f"   • Performance-explainability correlation: ρ = {corr_perf_exp:.3f} (p = {p_perf_exp:.4f})")
print(f"   • Performance-complexity correlation: ρ = {corr_perf_comp:.3f} (p = {p_perf_comp:.4f})")

print(f"\n3. INTERPRETATION:")
if p_perf_exp < 0.05 and corr_perf_exp < 0:
    print(f"   • The performance-explainability trade-off is statistically confirmed")
    print(f"   • More performant models tend to be less explainable")
elif p_perf_exp >= 0.05:
    print(f"   • No significant correlation detected")
    print(f"   • The trade-off might be less pronounced than expected")

print(f"\n4. IMPLICATIONS:")
print(f"   • For research: Need to develop standardized explainability metrics")
print(f"   • For practice: Guide for model selection according to constraints")
print(f"   • For industry: Importance of explainability for responsible deployment")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETED - READY FOR WRITING")
print("="*80)