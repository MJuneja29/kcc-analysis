"""
Saturation Analysis: Real Data Curve
Analyze how many queries can be solved by answering the top N% of unique question types.
Uses Leiden clustering results to demonstrate the Pareto Principle in farmer queries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

# Paths - use relative paths from script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
CLUSTERING_DIR = BASE_DIR / "outputs" / "leiden_clustering"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
DASHBOARD_DATA_DIR = BASE_DIR / "dashboards" / "src" / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_global_summary():
    """Load the global summary of all crops."""
    summary_path = CLUSTERING_DIR / "global_summary.csv"
    return pd.read_csv(summary_path)

def load_crop_clusters(crop_name):
    """Load cluster summary for a specific crop."""
    crop_dir = CLUSTERING_DIR / crop_name
    summary_path = crop_dir / "summary.csv"
    
    if not summary_path.exists():
        return None
    
    return pd.read_csv(summary_path)

def calculate_saturation_curve_per_crop(crop_name):
    """
    Calculate saturation curve for a single crop.
    Returns arrays of (pct_question_types, pct_queries_covered).
    """
    clusters = load_crop_clusters(crop_name)
    
    if clusters is None or len(clusters) == 0:
        return None, None
    
    # Sort by size (descending)
    clusters = clusters.sort_values('size', ascending=False).reset_index(drop=True)
    
    total_queries = clusters['size'].sum()
    total_clusters = len(clusters)
    
    # Calculate cumulative coverage
    cumulative_queries = clusters['size'].cumsum()
    cumulative_coverage = (cumulative_queries / total_queries) * 100
    
    # Calculate percentage of question types used
    pct_question_types = (np.arange(1, len(clusters) + 1) / total_clusters) * 100
    
    return pct_question_types, cumulative_coverage.values

def calculate_global_saturation_curve():
    """
    Calculate the global saturation curve across all crops.
    Aggregates all clusters from all crops and computes the overall efficiency.
    """
    global_summary = load_global_summary()
    
    all_clusters = []
    
    print("Loading cluster data from all crops...")
    for _, row in tqdm(global_summary.iterrows(), total=len(global_summary)):
        crop_name = row['crop']
        clusters = load_crop_clusters(crop_name)
        
        if clusters is not None:
            # Add crop name for reference
            clusters['crop'] = crop_name
            all_clusters.append(clusters[['crop', 'cluster_id', 'size', 'representative']])
    
    # Combine all clusters
    df_all = pd.concat(all_clusters, ignore_index=True)
    print(f"\nTotal clusters loaded: {len(df_all):,}")
    print(f"Total queries: {df_all['size'].sum():,}")
    
    # Sort by cluster size (descending)
    df_all = df_all.sort_values('size', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative metrics
    total_queries = df_all['size'].sum()
    total_clusters = len(df_all)
    
    df_all['cumulative_queries'] = df_all['size'].cumsum()
    df_all['pct_queries_covered'] = (df_all['cumulative_queries'] / total_queries) * 100
    df_all['pct_question_types'] = (np.arange(1, len(df_all) + 1) / total_clusters) * 100
    
    return df_all

def create_smoothed_curve(x, y, num_points=100):
    """Create a smoothed interpolation of the curve."""
    # Sample points evenly for smoothing
    indices = np.linspace(0, len(x) - 1, min(num_points, len(x))).astype(int)
    x_sampled = x[indices]
    y_sampled = y[indices]
    
    # Create smooth spline
    if len(x_sampled) > 3:
        x_smooth = np.linspace(x_sampled.min(), x_sampled.max(), 300)
        spl = make_interp_spline(x_sampled, y_sampled, k=3)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    else:
        return x_sampled, y_sampled

def plot_saturation_analysis(df_all, output_path):
    """
    Create the saturation analysis visualization matching the dashboard design.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set dark background
    fig.patch.set_facecolor('#1a2332')
    ax.set_facecolor('#1a2332')
    
    x = df_all['pct_question_types'].values
    y = df_all['pct_queries_covered'].values
    
    # Plot the real curve (green)
    ax.plot(x, y, color='#10b981', linewidth=2.5, label='Real Efficiency Curve', alpha=0.8)
    
    # Add grid
    ax.grid(True, alpha=0.15, color='gray', linestyle='-', linewidth=0.5)
    
    # Style axes
    ax.set_xlabel('% of Unique Question Types', fontsize=13, color='white', fontweight='500')
    ax.set_ylabel('% of Total Farmer Queries Solved', fontsize=13, color='white', fontweight='500')
    
    # Set limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Tick styling
    ax.tick_params(colors='gray', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#374151')
        spine.set_linewidth(1)
    
    # Add legend
    legend = ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    legend.get_frame().set_facecolor('#1f2937')
    legend.get_frame().set_edgecolor('#374151')
    
    # Add title and subtitle in the plot area
    title_text = "Saturation Analysis: Real Data Curve"
    subtitle_text = (
        f"Our analysis of {df_all['size'].sum():,.0f} queries reveals an aggressive saturation. "
        f"For every crop, a tiny set of \"vital\" questions repeats constantly."
    )
    insight_text = "The Insight: Extreme Pareto Principle"
    detail_text = (
        "The curve shows the actual cumulative coverage from our clustering analysis. "
        "By identifying and automating answers for just the **top 10%** of frequent question types, we\n"
        "effectively resolve the vast majority of all incoming farmer queries."
    )
    
    # Add text annotations
    fig.text(0.12, 0.96, title_text, fontsize=18, weight='bold', color='white', 
             ha='left', va='top')
    fig.text(0.12, 0.92, subtitle_text, fontsize=11, color='#9ca3af', 
             ha='left', va='top', wrap=True)
    fig.text(0.12, 0.88, insight_text, fontsize=13, weight='bold', color='white', 
             ha='left', va='top')
    fig.text(0.12, 0.85, detail_text, fontsize=10, color='#9ca3af', 
             ha='left', va='top')
    
    # Key statistics annotations
    # Find coverage at 10% and 20% of question types
    idx_10 = np.argmin(np.abs(x - 10))
    idx_20 = np.argmin(np.abs(x - 20))
    
    coverage_10 = y[idx_10]
    coverage_20 = y[idx_20]
    
    # Add annotation arrows
    ax.annotate(f'{coverage_10:.1f}% coverage\nat 10% of types', 
                xy=(10, coverage_10), xytext=(25, coverage_10 - 15),
                fontsize=10, color='#10b981', weight='bold',
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937', edgecolor='#10b981', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0, 1, 0.83])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1a2332')
    print(f"\n✓ Saturation plot saved to: {output_path}")
    plt.close()

def generate_saturation_data_for_dashboard(df_all):
    """Generate JSON data for the dashboard."""
    # Sample data points for the dashboard (every 0.5% of question types)
    sample_points = []
    
    x = df_all['pct_question_types'].values
    y = df_all['pct_queries_covered'].values
    
    # Sample at regular intervals
    for pct in np.arange(0, 100.5, 0.5):
        idx = np.argmin(np.abs(x - pct))
        sample_points.append({
            'pct_question_types': round(x[idx], 2),
            'pct_queries_covered': round(y[idx], 2)
        })
    
    # Add key statistics
    idx_10 = np.argmin(np.abs(x - 10))
    idx_20 = np.argmin(np.abs(x - 20))
    idx_30 = np.argmin(np.abs(x - 30))
    
    metadata = {
        'total_queries': int(df_all['size'].sum()),
        'total_question_types': int(len(df_all)),
        'coverage_at_10pct': round(y[idx_10], 2),
        'coverage_at_20pct': round(y[idx_20], 2),
        'coverage_at_30pct': round(y[idx_30], 2),
        'top_100_coverage': round(y[min(99, len(y)-1)], 2) if len(y) >= 100 else round(y[-1], 2)
    }
    
    result = {
        'metadata': metadata,
        'curve': sample_points
    }
    
    output_path = DASHBOARD_DATA_DIR / "saturationData.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Dashboard data saved to: {output_path}")
    
    return metadata

def print_key_statistics(df_all, metadata):
    """Print comprehensive statistics about the saturation analysis."""
    print("\n" + "="*80)
    print("SATURATION ANALYSIS - KEY INSIGHTS")
    print("="*80)
    
    total_queries = metadata['total_queries']
    total_types = metadata['total_question_types']
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total Farmer Queries: {total_queries:,}")
    print(f"   Total Unique Question Types (Clusters): {total_types:,}")
    print(f"   Average Queries per Type: {total_queries/total_types:.1f}")
    
    print(f"\n🎯 Pareto Principle Analysis:")
    print(f"   Top 10% of question types ({int(total_types*0.1):,} types) cover: {metadata['coverage_at_10pct']:.1f}% of queries")
    print(f"   Top 20% of question types ({int(total_types*0.2):,} types) cover: {metadata['coverage_at_20pct']:.1f}% of queries")
    print(f"   Top 30% of question types ({int(total_types*0.3):,} types) cover: {metadata['coverage_at_30pct']:.1f}% of queries")
    
    # Find how many types needed for various coverage levels
    x = df_all['pct_question_types'].values
    y = df_all['pct_queries_covered'].values
    
    for coverage_target in [50, 75, 80, 85, 90, 95]:
        idx = np.argmin(np.abs(y - coverage_target))
        pct_types = x[idx]
        num_types = int(total_types * pct_types / 100)
        print(f"   For {coverage_target}% coverage, need: {pct_types:.1f}% of types ({num_types:,} types)")
    
    print(f"\n💡 Strategic Insight:")
    print(f"   By creating standardized answers for just {int(total_types*0.1):,} question types,")
    print(f"   we can automatically resolve {metadata['coverage_at_10pct']:.1f}% of all incoming queries.")
    print(f"   This represents {int(total_queries * metadata['coverage_at_10pct'] / 100):,} queries!")
    
    # Top 10 question types
    print(f"\n🔝 Top 10 Question Types (by query volume):")
    top_10 = df_all.head(10)
    for i, row in top_10.iterrows():
        pct = (row['size'] / total_queries) * 100
        print(f"   {i+1:2d}. [{row['crop'][:20]:20s}] {row['representative'][:50]:50s} ({row['size']:,} queries, {pct:.2f}%)")
    
    print("\n" + "="*80)

def main():
    print("="*80)
    print("SATURATION ANALYSIS: REAL DATA CURVE")
    print("="*80)
    
    # Calculate global saturation curve
    df_all = calculate_global_saturation_curve()
    
    # Generate dashboard data
    metadata = generate_saturation_data_for_dashboard(df_all)
    
    # Create visualization
    output_path = OUTPUT_DIR / "saturation_analysis.png"
    plot_saturation_analysis(df_all, output_path)
    
    # Print statistics
    print_key_statistics(df_all, metadata)
    
    print("\n✅ Saturation analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
