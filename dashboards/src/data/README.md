# Data Directory

This directory contains the JSON data files used by the dashboard.

## Files

### `cropData.json`
Contains detailed information about 271 crops analyzed from the KCC dataset:
- Crop name and filename
- Total query count per crop
- Top FAQs with question text, count, and percentage

**Size**: ~198 KB  
**Format**: Array of crop objects

### `saturationData.json`
Contains saturation analysis metadata and curve data:
- Total queries: 7,435,441
- Total question types: 76,004
- Coverage percentages at 10%, 20%, 30%
- Saturation curve data points

**Size**: ~16 KB  
**Format**: Object with metadata and curve array

## Note

These files are excluded from version control via `.gitignore` due to their size. 
To use this dashboard, you'll need to generate these files using the data processing scripts.
