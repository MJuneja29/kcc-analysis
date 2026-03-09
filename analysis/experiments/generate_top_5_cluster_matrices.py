import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.pipeline.core.engine import ClusteringPipeline

def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    TOP_N_CROPS = 5             
    MAX_SAFE_QUERIES = 10000    
    
    EXCLUDED_QUERY_TYPES = [
        "Weather", "Sowing Time and Weather", "Market Information", 
        "Government Schemes", "Credit", "Loans", "Crop Insurance",
        "Noisy Data", "NA (Not Applicable)", "Training", 
        "Training and Exposure Visits", "Power, Roads etc.", "Soil Health Card"
    ]
    # =========================================================================
    
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / 'data' / 'processed' / 'kcc_master_dataset_remapped.csv'
    out_dir = script_dir.parent.parent / 'outputs' / 'production_agglomerative'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We will save to an Excel workbook to support multiple tabs
    out_excel = out_dir / 'top_5_algorithms_cluster_matrices.xlsx'
    
    # -------------------------------------------------------------------------
    # 1. LOAD AND FILTER
    # -------------------------------------------------------------------------
    print(f"Loading Master Dataset from: {data_path}")
    master_df = pd.read_csv(data_path, low_memory=False)
    master_df['Crop'] = master_df['Crop'].fillna('Unknown').astype(str)
    
    # Filter dynamics
    master_df = master_df[~master_df['QueryType'].isin(EXCLUDED_QUERY_TYPES)].copy()
    
    # Filter "Others"
    master_df = master_df[~master_df['Crop'].str.lower().str.strip().isin(['others', 'other'])]
    
    crop_counts = master_df.groupby('Crop').size().sort_values(ascending=False).reset_index(name='total_freq')
    top_crops = crop_counts.head(TOP_N_CROPS)['Crop'].tolist()
    
    metadata_cols = ['StateName', 'Season', 'QueryType']
    for col in metadata_cols:
        if col not in master_df.columns:
            master_df[col] = "Unknown"
            
    summary_tab_rows = []
    
    # We will store dataframes here so we can write them all to Excel at the end
    # Dict structure: crop_tabs_data[crop_name] = DataFrame
    crop_tabs_data = {crop: [] for crop in top_crops}

    # -------------------------------------------------------------------------
    # 2. RUN EXPERIMENTS
    # -------------------------------------------------------------------------
    for crop_name in top_crops:
        print(f"\n=============================================")
        print(f" PROCESSING CROP: {crop_name.upper()}")
        print(f"=============================================")
        
        df_crop = master_df[master_df['Crop'] == crop_name].copy()
        
        # Group to save memory
        df_grouped = df_crop.groupby(['QueryText'] + metadata_cols).size().reset_index(name='count')
        df_grouped = df_grouped.rename(columns={'QueryText': 'query_text'})
        df_grouped = df_grouped.sort_values('count', ascending=False).reset_index(drop=True)
        
        if len(df_grouped) > MAX_SAFE_QUERIES:
            df_grouped = df_grouped.head(MAX_SAFE_QUERIES)
            
        unique_vol = len(df_grouped)
        if unique_vol == 0: continue
        
        # Define the exact 5 configurations 
        experiments = [
            {
                "name": "Exp 1: OPTICS + Early Fusion",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.8, "algorithm": 'optics', 
                    "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 2: HDBSCAN + TF-IDF (Early Fusion)",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.5, "algorithm": 'hdbscan', 
                    "use_char_features": True, "use_length_feature": False, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 3: Agglomerative Mapping (Static K)",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.8, "algorithm": 'agglomerative', 
                    "algorithm_params": {'n_clusters': max(20, unique_vol // 20)}, 
                    "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 4: HDBSCAN + Multi-Feature Fusion (LLM + Length + Metadata)",
                "config": {
                    "model_name": 'all-MiniLM-L6-v2', "alpha": 0.5, "algorithm": 'hdbscan', 
                    "metadata_columns": ['StateName', 'Season'], "use_length_feature": True, "crop_name": crop_name
                }
            },
            {
                "name": "Exp 5: Dual-Pipeline (HDBSCAN -> Agglomerative)",
                "config": "DUAL_PIPELINE"
            }
        ]
        
        for exp in experiments:
            exp_name = exp['name']
            print(f"  >> Running {exp_name}")
            
            try:
                if exp['config'] == "DUAL_PIPELINE":
                    scout = ClusteringPipeline(model_name='all-MiniLM-L6-v2', alpha=0.8, algorithm='hdbscan', use_length_feature=True)
                    df_scout, _, _ = scout.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                    
                    dynamic_k = df_scout[df_scout['cluster_id'] != -1]['cluster_id'].nunique()
                    if dynamic_k < 2: dynamic_k = max(20, unique_vol // 20)
                    
                    main_pipe = ClusteringPipeline(model_name='all-MiniLM-L6-v2', alpha=0.8, algorithm='agglomerative', 
                                                   algorithm_params={'n_clusters': dynamic_k}, use_length_feature=True, crop_name=crop_name)
                    processed_df, _, _ = main_pipe.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                    
                else:
                    pipeline = ClusteringPipeline(**exp['config'])
                    processed_df, _, _ = pipeline.fit_predict(df_grouped, text_column='query_text', counts_column='count')
                
                # --- CALCULATE DETAILS ---
                valid_mask = processed_df['cleaned_text'].str.strip().astype(bool)
                valid_df = processed_df[valid_mask].copy()
                
                # Exclude noise for the detail metrics
                non_noise = valid_df[valid_df['cluster_id'] != -1].copy()
                total_non_noise_volume = non_noise['count'].sum()
                
                if total_non_noise_volume == 0:
                    continue
                    
                # Sort clusters by true volume
                cluster_vols = non_noise.groupby('cluster_id')['count'].sum().sort_values(ascending=False)
                
                cumulative_pct_tracker = 0.0
                ranked_cluster_index = 1
                
                for cid, vol in cluster_vols.items():
                    pct_of_crop = (vol / total_non_noise_volume) * 100
                    cumulative_pct_tracker += pct_of_crop
                    
                    # Extract the queries for this cluster
                    cluster_queries = non_noise[non_noise['cluster_id'] == cid].sort_values(by='count', ascending=False)
                    
                    # 1. Logic for the Main Summary Tab
                    top_5_raw_texts = cluster_queries['query_text'].head(5).tolist() # Get the 5 exact raw queries
                    
                    summary_tab_rows.append({
                        'Experiment': exp_name,
                        'crop': crop_name,
                        'cluster_id': ranked_cluster_index,
                        'queries_in_cluster': vol,
                        'pct_of_crop': round(pct_of_crop, 2),
                        'cumulative_pct': round(cumulative_pct_tracker, 2),
                        'top_5_queries': str(top_5_raw_texts)
                    })
                    
                    # 2. Logic for the Specific Crop Tab (Distinct query texts only)
                    # cluster_queries is already deduplicated since we grouped before clustering.
                    # Each row here is one unique question text — no repetition needed.
                    for _, row in cluster_queries.iterrows():
                        crop_tabs_data[crop_name].append({
                            'Experiment': exp_name,
                            'cluster_id': ranked_cluster_index,
                            'full_querytext': row['query_text']
                        })
                    
                    ranked_cluster_index += 1
                    
            except Exception as e:
                print(f"     [FAILED] {e}")
                
    # -------------------------------------------------------------------------
    # 3. WRITE TO MULTI-TAB EXCEL WORKBOOK
    # -------------------------------------------------------------------------
    print("\n=============================================")
    print(" ALL MAPPING COMPLETE. WRITING TO EXCEL...")
    
    with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
        
        # Write Main Summary Tab
        df_summary = pd.DataFrame(summary_tab_rows)
        # Reorder to match exact mentor specs
        df_summary = df_summary[['Experiment', 'crop', 'cluster_id', 'queries_in_cluster', 'pct_of_crop', 'cumulative_pct', 'top_5_queries']]
        df_summary.to_excel(writer, sheet_name='Main_Summary', index=False)
        
        # Write individual crop tabs
        for crop_name, row_data in crop_tabs_data.items():
            if not row_data: continue
            
            # Excel limits sheet names to 31 chars
            safe_sheet_name = str(crop_name)[:31].replace('/', '_').replace('\\', '_')
            
            df_crop_tab = pd.DataFrame(row_data)
            df_crop_tab = df_crop_tab[['Experiment', 'cluster_id', 'full_querytext']]
            df_crop_tab.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
    print(f" > SUCCESS! Generated workbook: {out_excel}")
    print("=============================================")

if __name__ == "__main__":
    main()
