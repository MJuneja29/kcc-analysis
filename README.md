# 🌾 KCC Analysis - Kisan Call Center Query Analysis

A comprehensive data analysis project that processes and analyzes **7.5 million farmer queries** from India's Kisan Call Center (KCC) system to identify patterns, FAQs, and automation opportunities.

## 📊 Project Overview

This project uses advanced NLP and clustering techniques to analyze farmer queries across 270+ crops, demonstrating that:
- **Top 100 question types** (0.13%) address **38.62%** of all queries
- **Top 10% of question types** address **89.21%** of all queries
- Enables targeted automation and standardization of responses

### Key Findings

- **Total Queries Analyzed**: 7,435,441
- **Unique Question Types**: 76,004
- **Crops Covered**: 273
- **Top Concerns**: Weather information, pest management, crop varieties, nutrient management

## 🏗️ Project Structure

```
KCC Analysis/
├── analysis/                  # Python analysis scripts
│   ├── improved_clustering.py # GPU-accelerated semantic clustering
│   ├── saturation_analysis.py # Pareto analysis of query distribution
│   └── sort_clusters.py       # Utility for sorting cluster results
│
├── data/
│   ├── aggregated/           # Small aggregated datasets (tracked in git)
│   │   ├── district_summary.csv
│   │   ├── monthly_time_series.csv
│   │   ├── query_type_by_year.csv
│   │   ├── state_summary.csv
│   │   └── top_50_crops.csv
│   ├── processed/            # Large processed datasets (git-ignored)
│   │   └── kcc_master_dataset_remapped.csv
│   └── raw/                  # Raw data files (git-ignored)
│
├── outputs/
│   ├── leiden_clustering/    # Individual crop clustering results (git-ignored)
│   │   └── [273 crop folders with mapping.csv, summary.csv, report.json]
│   ├── figures/              # Generated visualizations (git-ignored)
│   │   ├── crop_faqs/       # 303 crop visualization PNGs
│   │   ├── geographic/
│   │   ├── time_series/
│   │   └── saturation_analysis.png
│   └── SATURATION_ANALYSIS_README.md
│
├── dashboards/               # React web dashboard
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── data/           # Data files for dashboard (generated)
│   │   └── App.jsx
│   ├── public/
│   ├── package.json
│   ├── README.md           # Dashboard-specific README
│   └── docs/               # Deployment guides
│
├── assets/                  # Project assets
│   └── annam_logo.png
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher (for dashboard)
- **Git**: For cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "KCC Analysis"
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix/MacOS:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the dashboard**
   ```bash
   cd dashboards
   npm install
   ```

### Data Setup

⚠️ **Important**: Large data files are not included in the repository due to size constraints.

**Option 1: Download processed data**
- Contact the project maintainer for access to:
  - `data/processed/kcc_master_dataset_remapped.csv` (~50MB)
  - Pre-generated clustering results

**Option 2: Generate from scratch**
- Obtain raw KCC data
- Run clustering analysis (requires GPU for optimal performance)

## 📈 Usage

### Running Analysis Scripts

**1. Saturation Analysis**
```bash
python analysis/saturation_analysis.py
```
Generates:
- `outputs/figures/saturation_analysis.png`
- `dashboards/src/data/saturationData.json`

**2. Clustering Analysis**
```bash
python analysis/improved_clustering.py
```
Performs semantic clustering on farmer queries for a specific crop.

### Running the Dashboard

```bash
cd dashboards
npm run dev
```
Opens at: http://localhost:5174/

**Production build:**
```bash
npm run build
```

## 🔬 Technical Details

### Clustering Methodology

- **Model**: `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformers)
- **Algorithms**: HDBSCAN, K-Means, Leiden
- **Embeddings**: Hybrid dense (semantic) + sparse (TF-IDF)
- **Preprocessing**: Custom Hinglish stopword removal

### Dashboard Features

- Interactive crop explorer (271 crops)
- Top 10 crops with FAQ tables
- Saturation curve visualization
- Query distribution charts
- Real-time search and filtering

## 📊 Key Visualizations

1. **Saturation Curve**: Shows how many queries can be resolved by standardizing top N% of questions
2. **Crop FAQ Charts**: Individual visualizations for each crop's question distribution
3. **Geographic Analysis**: Query patterns by state/district
4. **Time Series**: Temporal trends in query types

## 🤝 Contributing

This is a team project. To contribute:

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📝 Documentation

- **Team Guide**: See [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) - **Complete pipeline architecture & dashboard details**
- **Dashboard**: See [dashboards/README.md](dashboards/README.md)
- **Saturation Analysis**: See [outputs/SATURATION_ANALYSIS_README.md](outputs/SATURATION_ANALYSIS_README.md)
- **Deployment**: See [dashboards/docs/](dashboards/docs/)

## 🎯 Impact & Applications

### Immediate Applications
- **FAQ Automation**: Create standardized answers for top 100 questions
- **Call Center Optimization**: Reduce manual query handling by 40%
- **Resource Planning**: Allocate experts based on query patterns

### Strategic Insights
- Weather information is the #1 farmer concern across all crops
- Pest/disease management queries have strong seasonal patterns
- Regional variations in query types suggest localized content needs

## 📧 Contact

For questions or data access, contact the project maintainer.

## 📄 License

[Specify your license here]

---

**Last Updated**: February 2026
**Data Period**: 2006 - Present
**Total Farmer Queries**: 7,435,441
