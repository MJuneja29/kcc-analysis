"""
visualize_pareto_coverage.py
-------------------------------
Reads the Main_Summary tab from generate_top_5_cluster_matrices.py output
and renders a clean two-panel figure:

  TOP:    Pareto saturation curves (one panel per crop)
  BOTTOM: Grouped bar chart — "How many clusters to reach 85% coverage?"
          This is the clearest way to compare algorithms directly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from pathlib import Path

# =========================================================================
# CONFIGURATION
# =========================================================================
EXCEL_PATH = Path(__file__).parent.parent.parent / 'outputs' / 'production_agglomerative' / 'top_5_algorithms_cluster_matrices.xlsx'
OUT_DIR    = Path(__file__).parent.parent.parent / 'outputs' / 'production_agglomerative'
TARGET_PCT = 85.0
# =========================================================================

def main():
    if not EXCEL_PATH.exists():
        print(f"[ERROR] Excel not found at: {EXCEL_PATH}")
        return

    print(f"Reading {EXCEL_PATH.name}...")
    df = pd.read_excel(EXCEL_PATH, sheet_name='Main_Summary')

    crops = sorted(df['crop'].unique())
    experiments = df['Experiment'].unique()

    # Shorten experiment names for the bar chart axis labels
    short_names = {
        'Exp 1: OPTICS + Early Fusion':                         'OPTICS',
        'Exp 2: HDBSCAN + TF-IDF (Early Fusion)':              'HDBSCAN\n+TF-IDF',
        'Exp 3: Agglomerative Mapping (Static K)':              'Agglom.\n(Static K)',
        'Exp 4: HDBSCAN + Multi-Feature Fusion (LLM + Length + Metadata)': 'HDBSCAN\n+Multi-Feature\nFusion',
        'Exp 5: Dual-Pipeline (HDBSCAN -> Agglomerative)':      'HDBSCAN\n→Agglomerative',
    }

    palette = sns.color_palette("tab10", n_colors=len(experiments))
    color_map = {exp: palette[i] for i, exp in enumerate(experiments)}

    # =========================================================================
    # PRE-COMPUTE: clusters needed to reach 85% per (crop, experiment)
    # =========================================================================
    bar_data = []
    for crop in crops:
        for exp in experiments:
            sub = df[(df['crop'] == crop) & (df['Experiment'] == exp)].sort_values('cluster_id')
            crossing = sub[sub['cumulative_pct'] >= TARGET_PCT]
            n_clusters = int(crossing.iloc[0]['cluster_id']) if not crossing.empty else None
            bar_data.append({'crop': crop, 'experiment': exp, 'clusters_to_85': n_clusters})

    bar_df = pd.DataFrame(bar_data).dropna(subset=['clusters_to_85'])
    bar_df['clusters_to_85'] = bar_df['clusters_to_85'].astype(int)

    # =========================================================================
    # FIGURE LAYOUT: Top row = curves, Bottom = bar chart
    # =========================================================================
    n_crops = len(crops)
    fig = plt.figure(figsize=(5 * n_crops, 14))
    gs = gridspec.GridSpec(2, n_crops, height_ratios=[1.4, 1], hspace=0.50, wspace=0.30)

    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------------------
    # TOP ROW: Saturation Curves (one per crop)
    # -------------------------------------------------------------------------
    for col_i, crop in enumerate(crops):
        ax = fig.add_subplot(gs[0, col_i])
        df_crop = df[df['crop'] == crop]

        ax.set_title(crop.replace(' ', '\n') if len(crop) > 12 else crop,
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('No. of Clusters', fontsize=9)
        if col_i == 0:
            ax.set_ylabel('Cumulative % Coverage', fontsize=9)
        ax.set_ylim(0, 102)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(labelsize=8)

        # 85% threshold line
        ax.axhline(y=TARGET_PCT, color='red', linestyle='--', linewidth=1.2, alpha=0.75)

        for exp in experiments:
            sub = df_crop[df_crop['Experiment'] == exp].sort_values('cluster_id')
            if sub.empty:
                continue
            ax.plot(sub['cluster_id'], sub['cumulative_pct'],
                    color=color_map[exp], linewidth=1.8, alpha=0.85)

            # Mark crossing with a small dot only
            cross = sub[sub['cumulative_pct'] >= TARGET_PCT]
            if not cross.empty:
                cx, cy = cross.iloc[0]['cluster_id'], cross.iloc[0]['cumulative_pct']
                ax.scatter([cx], [cy], color=color_map[exp], s=55, zorder=5, edgecolors='white', linewidth=0.5)

        ax.grid(axis='both', linestyle='--', alpha=0.35)

    # -------------------------------------------------------------------------
    # BOTTOM ROW: One grouped bar chart spanning all columns
    # -------------------------------------------------------------------------
    ax_bar = fig.add_subplot(gs[1, :])

    n_exp = len(experiments)
    x = np.arange(len(crops))
    bar_width = 0.8 / n_exp

    for i, exp in enumerate(experiments):
        sub = bar_df[bar_df['experiment'] == exp]
        values = [sub[sub['crop'] == c]['clusters_to_85'].values[0]
                  if len(sub[sub['crop'] == c]) > 0 else 0
                  for c in crops]
        offsets = x + (i - n_exp / 2) * bar_width + bar_width / 2
        bars = ax_bar.bar(offsets, values, width=bar_width * 0.88,
                          color=color_map[exp], alpha=0.85, label=short_names.get(exp, exp))

        # Value label on top of each bar
        for bar, val in zip(bars, values):
            if val > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            str(val), ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c.replace(' ', '\n') if len(c) > 12 else c for c in crops],
                           fontsize=9)
    ax_bar.set_ylabel('Clusters Needed to Reach 85% Coverage', fontsize=10)
    ax_bar.set_title('Algorithm Comparison — Fewer Bars = Better Compression of Queries',
                     fontsize=11, fontweight='bold', pad=10)
    ax_bar.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
    ax_bar.tick_params(axis='y', labelsize=8)

    # -------------------------------------------------------------------------
    # LEGEND & SAVING (Bottom Horizontal Fix)
    # -------------------------------------------------------------------------
    
    # Shared legend handles
    legend_handles = [mpatches.Patch(color=color_map[exp], label=short_names.get(exp, exp))
                      for exp in experiments]
    legend_handles.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                     linewidth=1.4, label='85% Threshold'))
    
    # FIX: Move legend to the bottom center and make it horizontal (ncol=6)
    legend = fig.legend(handles=legend_handles, loc='upper center', ncol=6,
                        bbox_to_anchor=(0.5, 0.06), fontsize=10, frameon=True)

    # Global title
    fig.suptitle('Pareto Analysis — KCC Farmer Query Clustering\n(Top 5 Algorithms × Top 5 Crops)',
                 fontsize=14, fontweight='bold', y=0.98)

    # FIX: Add margin at the bottom of the figure to make room for the legend
    fig.subplots_adjust(bottom=0.15)

    out_path = OUT_DIR / 'pareto_85pct_coverage.png'
    
    # Save with tight bounding box
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.4,
                bbox_extra_artists=[legend])
    print(f"\n  > Chart saved to: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()