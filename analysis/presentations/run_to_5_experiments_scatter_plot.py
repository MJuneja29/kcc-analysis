# -------------------------------------------------------------------------
# 5. GENERATE SCATTER PLOT
# -------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

script_dir = Path(__file__).parent
out_dir = script_dir.parent.parent / 'outputs' / 'benchmarks'
out_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.read_csv(out_dir / 'top_5_ultimate_experiments.csv')

try:
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
            
    # Scatter Plot: Noise Ratio on X (Lower is better), Silhouette on Y (Higher is better)
    # using 'jitter' by using stripplot OR simply adjust markers
    # Since seaborn scatterplot might overlap, we can use an Alpha or jitter
            
    ax = sns.scatterplot(
        data=results_df, 
        x="Noise_Ratio_Pct", 
        y="Silhouette_Score", 
        hue="Experiment", 
        style="Crop",
        s=200,          # Size of markers
        alpha=0.8,      # Slight transparency to see overlaps
        palette="tab10" # Distinct colors
    )
            
    # To further prevent overlap, we can add labels with a slight offset
    from adjustText import adjust_text
    texts = []
    for _, row in results_df.iterrows():
        # We can put a tiny crop initial to help see exactly what point is what
        texts.append(ax.text(row['Noise_Ratio_Pct'], row['Silhouette_Score'], row['Crop'][:2], fontsize=8, alpha=0.7))
                
    try:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    except Exception:
        pass # If adjustText library is not installed, fail silently and just leave text where it is
                
    plt.title('Algorithm Performance Benchmark\n(Top Left is Best: High Cohesion, Low Data Loss)', fontsize=16, pad=15)
    plt.xlabel('Noise Ratio % (Queries thrown away)', fontsize=12)
    plt.ylabel('Silhouette Score (Cluster Cohersion, -1 to 1)', fontsize=12)
            
    # Move legend outside to keep it extremely clean
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
            
    plot_path = out_dir / 'top_5_algorithm_scatter_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n  > Saved Scatter Plot to: {plot_path}")
    plt.close()
except Exception as e:
    print(f"  > Plotting failed: {e}")