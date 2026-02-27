import pandas as pd
import os
from pathlib import Path

# Define file paths - use relative paths from script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

input_file = PROJECT_ROOT / 'outputs' / 'leiden_clustering' / 'Paddy (Dhan)' / 'mapping.csv'
output_file = PROJECT_ROOT / 'outputs' / 'leiden_clustering' / 'Paddy (Dhan)' / 'mapping_sorted.csv'

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file not found at {input_file}")
    exit(1)

# Read the CSV file
try:
    df = pd.read_csv(input_file)
    print(f"Successfully read {len(df)} rows from {input_file}")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Sort by cluster_id
if 'cluster_id' in df.columns:
    df_sorted = df.sort_values(by=['cluster_id', 'count'], ascending=[True, False])
    print("Sorted DataFrame by cluster_id and count")
else:
    print("Error: 'cluster_id' column not found in DataFrame")
    exit(1)

# Save the sorted DataFrame
try:
    df_sorted.to_csv(output_file, index=False)
    print(f"Successfully saved sorted data to {output_file}")
except Exception as e:
    print(f"Error saving sorted CSV file: {e}")
    exit(1)
