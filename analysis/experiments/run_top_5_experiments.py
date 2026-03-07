import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.pipeline.core.engine import ClusteringPipeline
from analysis.pipeline.evaluation.evaluator import ClusteringEvaluator

def main():
    # =========================================================================
    # HARDWARE & GOAL CONFIGURATION
    # =========================================================================
    TOP_N_CROPS = 5             # Target top 5 crops
    MAX_SAFE_QUERIES = 10000    # RAM safety truncation threshold
    
    # Mentor's Request: Remove all dynamic/temporal query types so the AI 
    # focuses strictly on core agricultural practices
    EXCLUDED_QUERY_TYPES = [
        "Weather", "Sowing Time and Weather", "Market Information", 
        "Government Schemes", "Credit", "Loans", "Crop Insurance",
        "Noisy Data", "NA (Not Applicable)", "Training", 
        "Training and Exposure Visits", "Power, Roads etc.", "Soil Health Card"
    ]
    # =========================================================================
    
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / 'data' / 'processed' / 'kcc_master_dataset_remapped.csv'
    out_dir = script_dir.parent.parent / 'outputs' / 'benchmarks'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'top_5_ultimate_experiments.csv'
    
    # -------------------------------------------------------------------------
    # 1. LOAD AND FILTER MASTER DATASET
    # -------------------------------------------------------------------------
    print(f"Loading Master Dataset from: {data_path}")
    master_df = pd.read_csv(data_path, low_memory=False)
    master_df['Crop'] = master_df['Crop'].fillna('Unknown').astype(str)
    
    initial_len = len(master_df)
    master_df = master_df[~master_df['QueryType'].isin(EXCLUDED_QUERY_TYPES)].copy()
    print(f"  > [FILTER]: Dropped {initial_len - len(master_df)} dynamic/administrative rows.")
    
    # Drop "Others" crops 
    master_df = master_df[~master_df['Crop'].str.lower().str.strip().isin(['others', 'other'])]
    
    crop_counts = master_df.groupby('Crop').size().sort_values(ascending=False).reset_index(name='total_freq')
    top_crops = crop_counts.head(TOP_N_CROPS)['Crop'].tolist()
    
    # Ensure metadata columns exist for Early Fusion concatenation later
    metadata_cols = ['StateName', 'Season', 'QueryType']
    for col in metadata_cols:
        if col not in master_df.columns:
            master_df[col] = "Unknown"
            
    all_results = []
    
    # -------------------------------------------------------------------------
    # 2. RUN EXPERIMENTS PER CROP
    # -------------------------------------------------------------------------
    for crop_name in top_crops:
        print(f"\n=============================================")
        print(f" RUNNING BENCHMARKS FOR: {crop_name.upper()}")
        print(f"=============================================")
        
        df_crop = master_df[master_df['Crop'] == crop_name].copy()
        
        # Group identical queries to vastly decrease memory overhead for distance matrices
        df_grouped = df_crop.groupby(['QueryText'] + metadata_cols).size().reset_index(name='count')
        df_grouped = df_grouped.rename(columns={'QueryText': 'query_text'})
        df_grouped = df_grouped.sort_values('count', ascending=False).reset_index(drop=True)
        
        if len(df_grouped) > MAX_SAFE_QUERIES:
            print(f"  > Truncating to {MAX_SAFE_QUERIES} unique queries for local RAM safety.")
            df_grouped = df_grouped.head(MAX_SAFE_QUERIES)
            
        unique_vol = len(df_grouped)
        if unique_vol == 0:
            continue
            
        print(f"  > Total Unique Agricultural Queries To Process: {unique_vol}")
            
        # Define the exact 5 configurations tested historically.
        # Note: ALL 5 experiments use "Early Fusion" (either blending LLM sentence embeddings 
        # with query length, or with raw TF-IDF vocabulary weights) via the `alpha` parameter.
        experiments = [
            {
                "name": "Exp 1: OPTICS + Early Fusion",
                "desc": "Uses OPTICS algorithm for highly varying language density. Blends LLM vectors with text-length features via Early Fusion.",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.8, "algorithm": 'optics', 
                    "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 2: HDBSCAN + TF-IDF (Early Fusion)",
                "desc": "Fuses dense semantic LLM vectors with sparse vocabulary TF-IDF matrices before running UMAP/HDBSCAN.",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.5, "algorithm": 'hdbscan', 
                    "use_char_features": True, "use_length_feature": False, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 3: Agglomerative Mapping (Static K)",
                "desc": "Forces every query into a bucket (0% noise) using a static pre-determined cluster count proportional to dataset size.",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.8, "algorithm": 'agglomerative', 
                    "algorithm_params": {'n_clusters': max(20, unique_vol // 20)}, # Set static K based on crop scale
                    "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 4: God Mode (HDBSCAN + Length + Metadata Fusion)",
                "desc": "The ultimate Early Fusion: Concatenates LLM sentence vectors, sentence length traits, and State/Season one-hot metadata.",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.5, "algorithm": 'hdbscan', 
                    "metadata_columns": ['StateName', 'Season'], "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 5: Dual-Pipeline (HDBSCAN -> Agglomerative)",
                "desc": "Production logic: Uses HDBSCAN to magically 'scout' the perfect K value, then uses Agglomerative to map it flawlessly.",
                "config": "DUAL_PIPELINE" # Custom orchestration flow handled below
            }
        ]
        
        # -------------------------------------------------------------------------
        # 3. EXECUTE THE 5 STRATEGIES
        # -------------------------------------------------------------------------
        for exp in experiments:
            exp_name = exp['name']
            print(f"\n  >> Starting: {exp_name}")
            print(f"     Description: {exp['desc']}")
            
            t0 = time.time()
            try:
                # Custom handler for the #1 ranked Dual-Pipeline
                if exp['config'] == "DUAL_PIPELINE":
                    # Phase 1: Scout
                    scout = ClusteringPipeline(model_name='all-MiniLM-L6-v2', alpha=0.8, algorithm='hdbscan', use_length_feature=True)
                    df_scout, _, _ = scout.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                    
                    # Read dynamic K value without human intervention
                    dynamic_k = df_scout[df_scout['cluster_id'] != -1]['cluster_id'].nunique()
                    if dynamic_k < 2: dynamic_k = max(20, unique_vol // 20)
                    
                    # Phase 2: Map
                    main_pipe = ClusteringPipeline(model_name='all-MiniLM-L6-v2', alpha=0.8, algorithm='agglomerative', 
                                                   algorithm_params={'n_clusters': dynamic_k}, use_length_feature=True, crop_name=crop_name)
                    processed_df, reduced_features, _ = main_pipe.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                    
                # Standard Engine Hook for standalone methods
                else:
                    pipeline = ClusteringPipeline(**exp['config'])
                    processed_df, reduced_features, _ = pipeline.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                    
                duration_sec = time.time() - t0
                
                # Strip blank texts and evaluate mathematically
                valid_mask = processed_df['cleaned_text'].str.strip().astype(bool)
                valid_df = processed_df[valid_mask]
                labels = valid_df['cluster_id'].values
                
                # Fetch scoring mechanisms (Silhouette, Davies-Bouldin, Coverage Matrix)
                metrics = ClusteringEvaluator.evaluate(valid_df, labels, reduced_features, counts_column='count')
                
                all_results.append({
                    'Crop': crop_name,
                    'Experiment': exp_name,
                    'Description': exp['desc'],
                    'Queries_Processed': unique_vol,
                    'Silhouette_Score': round(metrics.get('silhouette', 0), 4),
                    'Noise_Ratio_Pct': round(metrics.get('noise_ratio_pct', 0), 2),
                    'Top_10_Pct_Coverage': round(metrics.get('top_10_pct_coverage', 0), 2),
                    'Total_Clusters': metrics.get('total_clusters', 0),
                    'Execution_Time_Sec': round(duration_sec, 2)
                })
                
                print(f"     [SUCCESS] Time: {round(duration_sec, 2)}s | Silhouette: {round(metrics.get('silhouette', 0), 2)} | Noise: {round(metrics.get('noise_ratio_pct', 0), 2)}%")
                
            except Exception as e:
                print(f"     [FAILED] Algorithm Crashed contextually: {e}")
                
    # -------------------------------------------------------------------------
    # 4. SAVE AND EXPORT THE SCORECARD
    # -------------------------------------------------------------------------
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Sort output by Crop name alphabetically, and then rank by Highest Silhouette Score 
        results_df = results_df.sort_values(by=['Crop', 'Silhouette_Score'], ascending=[True, False])
        results_df.to_csv(out_csv, index=False)
        
        print(f"\n=============================================")
        print(f" ALL 5 EXPERIMENTS FINISHED FOR TOP 5 CROPS.")
        print(f" The scorecard matrix was written to: \n {out_csv}")
        print(f"=============================================")

if __name__ == "__main__":
    main()
