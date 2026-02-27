# KCC Analysis - Team Technical Guide

**Version**: 1.0  
**Last Updated**: February 27, 2026  
**Maintainers**: KCC Analysis Team

---

## Table of Contents

1. [Pipeline Architecture Overview](#pipeline-architecture-overview)
2. [Data Flow & Processing](#data-flow--processing)
3. [Analysis Pipeline Details](#analysis-pipeline-details)
4. [Dashboard Architecture](#dashboard-architecture)
5. [API & Data Contracts](#api--data-contracts)
6. [Development Workflows](#development-workflows)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pipeline Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KCC ANALYSIS PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Raw Data    │  7.5M+ Farmer Queries (CSV format)
│  Collection  │  - Query text (Hinglish)
└──────┬───────┘  - Crop name
       │          - Date, Location
       │          - Query type
       ▼
┌──────────────┐
│  Data Pre-   │  Python: Pandas
│  Processing  │  - Text normalization
└──────┬───────┘  - Crop name standardization
       │          - Deduplication
       │          - Feature extraction
       ▼
┌──────────────┐
│  NLP &       │  Python: Sentence Transformers
│  Embedding   │  - Multilingual model (paraphrase-mpnet)
└──────┬───────┘  - 768-dim embeddings
       │          - Hinglish support
       │
       ▼
┌──────────────┐
│  Semantic    │  Python: HDBSCAN, K-Means, Leiden
│  Clustering  │  - Hybrid embeddings (Dense + Sparse)
└──────┬───────┘  - Per-crop clustering
       │          - 273 crop analyses
       │
       ▼
┌──────────────┐
│  Saturation  │  Python: NumPy, Scipy
│  Analysis    │  - Pareto analysis
└──────┬───────┘  - Coverage curves
       │          - JSON export
       │
       ▼
┌──────────────┐
│  Dashboard   │  React + Vite
│  Rendering   │  - Chart.js visualizations
└──────────────┘  - Interactive UI
                   - Responsive design
```

### Technology Stack

| Layer | Technologies | Purpose |
|-------|-------------|---------|
| **Data Processing** | Python 3.8+, Pandas, NumPy | ETL, aggregation |
| **NLP/ML** | Sentence Transformers, HDBSCAN, UMAP, scikit-learn | Semantic analysis, clustering |
| **Analysis** | SciPy, Matplotlib, Seaborn | Statistical analysis, visualization |
| **Backend Data** | JSON, CSV | Data storage & exchange |
| **Frontend** | React 18, Vite, JavaScript ES6+ | Interactive UI |
| **Visualization** | Chart.js, React-ChartJS-2 | Charts & graphs |
| **Deployment** | Vercel, Render, Docker | Hosting & CI/CD |

---

## Data Flow & Processing

### 1. Input Data Structure

**File**: `data/processed/kcc_master_dataset_remapped.csv`

```csv
QueryID, QueryText, Crop, District, State, Date, QueryType, ...
1, "gehu me kida laga hai", "Wheat", "Lucknow", "UP", "2023-01-15", "Pest", ...
```

**Key Fields**:
- `QueryText`: Original farmer question (Hinglish/Hindi/English mix)
- `Crop`: Standardized crop name (273 unique crops)
- `QueryType`: Category (Pest, Disease, Nutrient, etc.)
- Geospatial: State, District
- Temporal: Date, Year, Month

### 2. Data Aggregation

**Scripts**: Various ETL processes generate:

```
data/aggregated/
├── district_summary.csv      # Query counts by district
├── monthly_time_series.csv   # Temporal trends
├── query_type_by_year.csv    # Category evolution
├── state_summary.csv         # State-level statistics
└── top_50_crops.csv          # Most queried crops
```

### 3. Processing Pipeline

```python
# Pseudocode for main processing flow

1. Load raw data
   └─> pd.read_csv('kcc_master_dataset_remapped.csv')

2. For each crop:
   ├─> Filter queries by crop
   ├─> Preprocess text (remove stopwords, normalize)
   ├─> Generate embeddings (768-dim vectors)
   ├─> Generate TF-IDF features (sparse representation)
   ├─> Combine embeddings (α * dense + (1-α) * sparse)
   ├─> Cluster similar questions
   ├─> Identify representative questions per cluster
   └─> Export results (mapping.csv, summary.csv, report.json)

3. Global saturation analysis:
   ├─> Aggregate all clusters across crops
   ├─> Sort by cluster size (query count)
   ├─> Calculate cumulative coverage
   ├─> Generate saturation curve
   └─> Export (saturationData.json, saturation_analysis.png)
```

---

## Analysis Pipeline Details

### Component 1: Text Preprocessing

**File**: `analysis/improved_clustering.py`

**Function**: `preprocess_text(text)`

```python
# Preprocessing steps:
1. Lowercase conversion
2. Special character removal (keep only alphabets)
3. Extra whitespace removal
4. Hinglish stopword filtering (200+ words)
   - Generic: the, is, are, in, on
   - Hinglish: mein, ko, ka, ki, hai, kaise
   - Agriculture: crop, kisan, farm, seed
```

**Custom Stopwords**: Carefully curated to preserve domain-specific terms while removing noise.

### Component 2: Hybrid Embedding Generation

**Architecture**: Dense (Semantic) + Sparse (Lexical)

#### Dense Embeddings
- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Dimension**: 768
- **Language Support**: Multilingual (supports Hinglish)
- **Captures**: Semantic meaning, context, intent

#### Sparse Embeddings (TF-IDF)
- **Vectorizer**: scikit-learn TfidfVectorizer
- **Max Features**: 1000
- **Parameters**: `max_df=0.95, min_df=2`
- **Captures**: Keyword importance, lexical patterns

#### Hybrid Distance Calculation
```python
hybrid_distance = α × cosine_distance(dense_embeddings) + 
                  (1-α) × jaccard_distance(tfidf_features)

# Default: α = 0.5 (equal weight)
```

**Why Hybrid?**
- Dense: Handles synonyms, paraphrasing ("kida" vs "insect")
- Sparse: Catches exact keyword matches, domain terms
- Combined: Best of both worlds

### Component 3: Clustering Algorithms

**Primary**: K-Means (for consistency)  
**Alternatives**: HDBSCAN, Leiden (for experimentation)

#### K-Means Configuration
```python
n_clusters = 2000  # Fixed for consistency across crops
algorithm = 'auto'
n_init = 10
max_iter = 300
```

**Why 2000 clusters?**
- Balance between granularity and interpretability
- Provides ~85% coverage with top 15-30% of clusters
- Manageable for manual review

#### Output Structure
```
outputs/leiden_clustering/[Crop Name]/
├── mapping.csv          # Every query mapped to cluster
│   Columns: query_text, cluster_id, count
│
├── summary.csv          # Cluster statistics
│   Columns: cluster_id, size, unique_queries, representative, percentage
│
├── report.json          # Metadata
│   {crop, total_queries, unique_queries, total_clusters, ...}
│
└── cluster_viz.html     # Interactive visualization (optional)
```

### Component 4: Saturation Analysis

**File**: `analysis/saturation_analysis.py`

**Purpose**: Demonstrate Pareto Principle in farmer queries

#### Algorithm
```python
1. Load all crop clustering results (273 crops)
2. Aggregate all clusters globally
3. Sort by cluster size (descending)
4. Calculate cumulative coverage:
   
   For each cluster i:
   - cumulative_queries = sum(cluster_sizes[0:i])
   - coverage_pct = cumulative_queries / total_queries
   - types_pct = i / total_clusters
   
5. Generate curve: (types_pct, coverage_pct)
6. Export for visualization
```

#### Key Metrics Calculated
```json
{
  "total_queries": 7435441,
  "total_question_types": 76004,
  "top_100_coverage": 38.62,      // % queries covered by top 100 types
  "coverage_at_10pct": 89.21,      // % queries covered by top 10% types
  "coverage_at_20pct": 95.26,
  "coverage_at_30pct": 97.69
}
```

#### Output Files
- `dashboards/src/data/saturationData.json` - For dashboard
- `outputs/figures/saturation_analysis.png` - Static visualization

---

## Dashboard Architecture

### React Application Structure

```
dashboards/
├── public/
│   └── figures/
│       └── crop_faqs/          # 303 crop visualizations (PNG)
│
├── src/
│   ├── components/
│   │   ├── Header.jsx          # Dashboard header
│   │   ├── StatsCards.jsx      # Key metrics display
│   │   ├── Top100Insight.jsx   # Top 100 FAQs impact section
│   │   ├── SaturationChart.jsx # Pareto curve visualization
│   │   ├── TopCrops.jsx        # Top 10 crops with FAQs
│   │   ├── DistributionChart.jsx # Query distribution by crop
│   │   └── CropExplorer.jsx    # Searchable crop database
│   │
│   ├── data/
│   │   ├── cropData.json       # 271 crops with FAQs (~200KB)
│   │   └── saturationData.json # Saturation metrics (~17KB)
│   │
│   ├── App.jsx                 # Main application component
│   ├── App.css                 # Global styles
│   └── main.jsx                # Entry point
│
├── package.json
├── vite.config.js
└── index.html
```

### Component Details

#### 1. Header Component
**File**: `src/components/Header.jsx`

**Purpose**: Display dashboard title and subtitle

**Props**: None

**Renders**:
```jsx
<div className="header">
  <h1>🌾 UP Crop FAQ Dashboard</h1>
  <p>Analysis of 7.4 Million Farmer Queries | Efficiency Insights</p>
</div>
```

---

#### 2. StatsCards Component
**File**: `src/components/StatsCards.jsx`

**Purpose**: Display key metrics in card format

**Props**:
- `totalQueries` (number): Total queries analyzed
- `totalCrops` (number): Number of crops
- `totalQuestionTypes` (number): Unique question types
- `coverageAt10Pct` (number): Coverage percentage

**Layout**: 4 cards in grid layout

**Features**:
- Number formatting (7.4M, 76k)
- Responsive grid
- Hover effects

---

#### 3. Top100Insight Component
**File**: `src/components/Top100Insight.jsx`

**Purpose**: Visualize impact of addressing top 100 FAQs

**Props**:
- `saturationData` (object): Saturation analysis data

**Features**:
- **Doughnut Chart**: Shows 38.62% vs 61.38% split
- **Insight Box**: Quick win opportunities
- **Statistics Grid**: Coverage metrics
- **Chart.js Integration**: Interactive tooltips

**Data Visualization**:
```javascript
{
  labels: ['Top 100 FAQs', 'Remaining Questions'],
  datasets: [{
    data: [38.62, 61.38],
    backgroundColor: ['#10b981', '#374151']
  }]
}
```

**Key Insights Displayed**:
- ✅ Queries automatically resolved: 2,871,235
- ✅ Workload reduction: 38.62%
- ✅ Instant response capability
- ✅ Agent time freed for complex queries

---

#### 4. SaturationChart Component
**File**: `src/components/SaturationChart.jsx`

**Purpose**: Display Pareto curve showing efficiency gains

**Props**:
- `saturationData` (object): Full saturation analysis

**Features**:
- **Line Chart**: Real efficiency curve (green)
- **Trend Line**: Smoothed curve (cyan dashed)
- **Interactive Tooltips**: Hover for exact values
- **Statistics Cards**: 10%, 20%, 30% coverage milestones
- **Insight Boxes**: Strategic recommendations

**Chart Configuration**:
```javascript
{
  type: 'line',
  data: {
    labels: [0, 0.5, 1.0, ..., 100],  // % of question types
    datasets: [
      {
        label: 'Real Efficiency Curve',
        data: coverage_percentages,
        borderColor: '#10b981',
        tension: 0.4
      }
    ]
  }
}
```

**Key Visualizations**:
- X-axis: % of Question Types (0-100%)
- Y-axis: % of Queries Resolved (0-100%)
- Goal: Show steep initial curve (Pareto principle)

---

#### 5. TopCrops Component
**File**: `src/components/TopCrops.jsx`

**Purpose**: Display top 10 most queried crops with FAQs

**Props**:
- `crops` (array): Array of crop objects (top 10)

**Structure for Each Crop**:
```jsx
<div className="crop-section">
  <h3>{crop_name}</h3>
  <p>{query_count.toLocaleString()} queries</p>
  <img src={crop_visualization_png} />
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Question Theme</th>
        <th>Queries</th>
        <th>%</th>
      </tr>
    </thead>
    <tbody>
      {top_5_faqs.map(faq => <tr>...</tr>)}
    </tbody>
  </table>
</div>
```

**Features**:
- Query count formatting (1.2M, 500k)
- Percentage display per FAQ
- Visual aids (crop images)
- Responsive layout

---

#### 6. DistributionChart Component
**File**: `src/components/DistributionChart.jsx`

**Purpose**: Show crop distribution by query volume brackets

**Props**:
- `crops` (array): All 271 crops

**Visualization**: Horizontal bar chart

**Categories**:
```javascript
{
  '1M+': crops with > 1,000,000 queries,
  '500k-1M': crops with 500,000-999,999 queries,
  '100k-500k': crops with 100,000-499,999 queries,
  '50k-100k': crops with 50,000-99,999 queries,
  '10k-50k': crops with 10,000-49,999 queries,
  '<10k': crops with < 10,000 queries
}
```

**Chart.js Config**:
```javascript
{
  type: 'bar',
  indexAxis: 'y',  // Horizontal bars
  scales: {
    x: { title: 'Number of Crops' },
    y: { title: 'Query Volume Range' }
  }
}
```

---

#### 7. CropExplorer Component
**File**: `src/components/CropExplorer.jsx`

**Purpose**: Searchable database of all 271 crops

**Props**:
- `crops` (array): All crop objects

**Features**:

1. **Search Bar**
   - Real-time filtering
   - Case-insensitive
   - Searches crop names

2. **Crop Grid**
   - Card layout
   - Shows: Name, query count, thumbnail
   - Click to expand

3. **Detailed View (Modal)**
   - Full crop name
   - Total queries
   - Complete FAQ table (all themes)
   - Large visualization image
   - Close button

**State Management**:
```javascript
const [searchTerm, setSearchTerm] = useState('')
const [selectedCrop, setSelectedCrop] = useState(null)
const filteredCrops = crops.filter(crop => 
  crop.name.toLowerCase().includes(searchTerm.toLowerCase())
)
```

**FAQ Table Structure**:
```jsx
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Question Theme</th>
      <th>Queries</th>
      <th>Coverage %</th>
    </tr>
  </thead>
  <tbody>
    {crop.faqs.map((faq, index) => (
      <tr key={index}>
        <td>{index + 1}</td>
        <td>{faq.question}</td>
        <td>{faq.count.toLocaleString()}</td>
        <td>{faq.percentage}%</td>
      </tr>
    ))}
  </tbody>
</table>
```

---

### Data Contracts

#### cropData.json Structure
```json
[
  {
    "name": "Wheat",
    "filename": "Wheat",
    "queries": 992538,
    "faqs": [
      {
        "question": "farmer want information about weather",
        "count": 180006,
        "percentage": 18.14
      },
      {
        "question": "information for weather",
        "count": 47753,
        "percentage": 4.81
      },
      // ... more FAQs (typically 20-50 per crop)
    ]
  },
  // ... 270 more crops
]
```

#### saturationData.json Structure
```json
{
  "metadata": {
    "total_queries": 7435441,
    "total_question_types": 76004,
    "top_100_coverage": 38.62,
    "coverage_at_10pct": 89.21,
    "coverage_at_20pct": 95.26,
    "coverage_at_30pct": 97.69,
    "crops_analyzed": 273,
    "generated_date": "2024-02-27"
  },
  "curve_data": [
    {"pct_types": 0.0, "pct_coverage": 0.0},
    {"pct_types": 0.5, "pct_coverage": 15.2},
    {"pct_types": 1.0, "pct_coverage": 25.4},
    // ... 201 data points total (0-100% in 0.5% steps)
  ],
  "top_10_questions": [
    {
      "crop": "Others",
      "question": "about weather information",
      "count": 456308,
      "percentage": 6.14
    },
    // ... top 10
  ]
}
```

---

## API & Data Contracts

### File Locations

| Data File | Location | Size | Purpose | Generated By |
|-----------|----------|------|---------|--------------|
| cropData.json | `dashboards/src/data/` | ~200KB | All crop FAQs | Manual/Script |
| saturationData.json | `dashboards/src/data/` | ~17KB | Saturation metrics | `saturation_analysis.py` |
| Crop PNGs | `dashboards/public/figures/crop_faqs/` | ~50MB total | Visualizations | Analysis scripts |
| mapping.csv | `outputs/leiden_clustering/[Crop]/` | Varies | Query-to-cluster mapping | `improved_clustering.py` |
| summary.csv | `outputs/leiden_clustering/[Crop]/` | ~50-200KB | Cluster summaries | `improved_clustering.py` |
| report.json | `outputs/leiden_clustering/[Crop]/` | <1KB | Metadata | `improved_clustering.py` |

### Data Generation Scripts

```bash
# Generate clustering results for a crop
python analysis/improved_clustering.py
# → Creates mapping.csv, summary.csv, report.json

# Generate saturation analysis
python analysis/saturation_analysis.py
# → Creates saturationData.json, saturation_analysis.png

# Generate cropData.json (you'll need to create this script)
python scripts/generate_crop_data.py
# → Aggregates all crop summaries into single JSON
```

---

## Development Workflows

### Workflow 1: Adding a New Crop Analysis

```bash
# 1. Update crop name in clustering script
cd analysis
nano improved_clustering.py
# Change INPUT_FILE path to new crop

# 2. Run clustering
python improved_clustering.py

# 3. Verify outputs
ls ../outputs/leiden_clustering/[NewCrop]/
# Should see: mapping.csv, summary.csv, report.json

# 4. Re-run saturation analysis to include new crop
python saturation_analysis.py

# 5. Update dashboard data (if needed)
# Regenerate cropData.json to include new crop

# 6. Test dashboard
cd ../dashboards
npm run dev
```

### Workflow 2: Updating Dashboard Components

```bash
# 1. Create/modify component
cd dashboards/src/components
nano MyNewComponent.jsx

# 2. Import in App.jsx
nano ../App.jsx
# Add: import MyNewComponent from './components/MyNewComponent'

# 3. Test locally
npm run dev

# 4. Build for production
npm run build

# 5. Preview production build
npm run preview

# 6. Deploy (Vercel)
git push origin main
# Vercel auto-deploys from main branch
```

### Workflow 3: Updating Analysis Parameters

**Scenario**: Change clustering parameters

```python
# In analysis/improved_clustering.py

# Change these parameters:
ALPHA = 0.7  # Currently 0.5 - increase for more semantic weight
n_clusters = 3000  # Currently 2000 - more granularity
max_features = 2000  # Currently 1000 - more TF-IDF features

# Re-run for affected crops
python improved_clustering.py
```

**Impacts**:
- More clusters = finer-grained questions, longer processing
- Higher alpha = more semantic similarity, less keyword matching
- More TF-IDF features = better keyword capture, more memory

---

## Troubleshooting Guide

### Issue 1: Clustering Script Fails with Memory Error

**Symptom**: `MemoryError` during distance calculation

**Solution**:
```python
# In improved_clustering.py, reduce data size:
# Option A: Sample data
df = df.sample(n=100000)  # Use subset for testing

# Option B: Reduce features
max_features = 500  # Lower from 1000

# Option C: Use sparse distance only
# Comment out dense distance calculation
```

### Issue 2: Dashboard Shows "Loading..." Forever

**Symptom**: Dashboard loads but shows spinner indefinitely

**Root Causes**:
1. Missing data files (cropData.json, saturationData.json)
2. Malformed JSON
3. Network error (if loading from URL)

**Solutions**:
```bash
# Check if data files exist
ls dashboards/src/data/
# Should see: cropData.json, saturationData.json

# Validate JSON syntax
cd dashboards/src/data
python -m json.tool cropData.json > /dev/null
python -m json.tool saturationData.json > /dev/null

# Check browser console for errors
# Open DevTools → Console tab
```

### Issue 3: Saturation Analysis Shows Incorrect Percentages

**Symptom**: Coverage percentages don't match expected Pareto distribution

**Debug Steps**:
```python
# Add debug prints in saturation_analysis.py

print(f"Total queries: {df['size'].sum()}")
print(f"Total clusters: {len(df)}")
print(f"Top 10 clusters: {df.head(10)['size'].sum()}")
print(f"Coverage: {df.head(10)['size'].sum() / df['size'].sum() * 100:.2f}%")
```

**Common Issues**:
- Missing crop folders in leiden_clustering/
- Corrupted summary.csv files
- Duplicate cluster IDs

### Issue 4: Charts Not Rendering

**Symptom**: Blank spaces where charts should be

**Causes**:
1. Chart.js not registered
2. Data format mismatch
3. CSS display issues

**Solutions**:
```javascript
// In component file, ensure Chart.js is registered
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    // ... all required elements
} from 'chart.js'

ChartJS.register(
    CategoryScale,
    LinearScale,
    // ... register all
)

// Check data format
console.log('Chart data:', data)
// Should match Chart.js expected format

// Check CSS
.chart-container {
  height: 400px;  // Must have explicit height
  width: 100%;
}
```

### Issue 5: Deployment Fails on Vercel

**Symptom**: Build succeeds locally but fails on Vercel

**Common Issues**:

1. **Missing environment variables**
   ```bash
   # In Vercel dashboard, add:
   NODE_VERSION=18
   ```

2. **Data files too large**
   ```javascript
   // In vite.config.js, exclude large files:
   export default defineConfig({
     build: {
       rollupOptions: {
         external: ['./src/data/largefile.json']
       }
     }
   })
   ```

3. **Node version mismatch**
   ```json
   // In package.json, specify:
   "engines": {
     "node": ">=18.0.0"
   }
   ```

### Issue 6: Slow Dashboard Performance

**Symptom**: Dashboard takes long to load/render

**Optimizations**:

1. **Lazy load crop images**
   ```jsx
   <img loading="lazy" src={cropImage} />
   ```

2. **Virtualize long lists**
   ```bash
   npm install react-window
   ```
   ```jsx
   import { FixedSizeList } from 'react-window'
   // Use for CropExplorer with 271 crops
   ```

3. **Code splitting**
   ```jsx
   const CropExplorer = lazy(() => import('./components/CropExplorer'))
   ```

4. **Optimize data files**
   - Compress JSON (gzip)
   - Remove unnecessary fields
   - Use pagination for large datasets

---

## Performance Benchmarks

### Analysis Pipeline

| Task | Data Size | Time | Hardware |
|------|-----------|------|----------|
| Text Preprocessing | 100k queries | ~30s | 8-core CPU |
| Dense Embeddings | 100k queries | ~15min | GPU (T4) |
| Dense Embeddings | 100k queries | ~2hr | CPU only |
| TF-IDF Vectorization | 100k queries | ~10s | CPU |
| Distance Calculation | 100k queries | ~5min | CPU (8-core) |
| HDBSCAN Clustering | 100k queries | ~3min | CPU |
| K-Means Clustering | 100k queries | ~1min | CPU |
| Full Pipeline (1 crop) | ~100k queries | ~2hr | GPU + 8-core CPU |

### Dashboard

| Metric | Value | Target |
|--------|-------|--------|
| Initial Load Time | 2.5s | <3s |
| Time to Interactive | 3.2s | <5s |
| Bundle Size | 450KB (gzipped) | <500KB |
| Lighthouse Score | 92/100 | >90 |
| First Contentful Paint | 1.8s | <2s |

---

## Best Practices

### For Data Scientists

1. **Always use relative paths** in scripts
2. **Test on sample data** before full run
3. **Document parameter changes** in git commits
4. **Validate outputs** before pushing
5. **Use virtual environments** to avoid conflicts

### For Frontend Developers

1. **Follow React best practices** (hooks, functional components)
2. **Optimize images** before adding to public/
3. **Test on multiple devices** (mobile, tablet, desktop)
4. **Use semantic HTML** for accessibility
5. **Profile performance** with React DevTools

### For DevOps

1. **Monitor build times** on CI/CD
2. **Set up branch protection** for main
3. **Configure auto-deployment** (Vercel/Render)
4. **Use environment variables** for configs
5. **Set up error tracking** (Sentry, etc.)

---

## Future Enhancements

### Planned Features

1. **Real-time Query Analysis**
   - Live data ingestion pipeline
   - Incremental clustering updates
   - Dashboard auto-refresh

2. **Multi-language Support**
   - Hindi UI translation
   - Regional language support
   - Language detection

3. **Advanced Filtering**
   - Filter by date range
   - Filter by region
   - Filter by query type

4. **Export Functionality**
   - PDF reports
   - CSV exports
   - API access

5. **Admin Dashboard**
   - User management
   - Analytics tracking
   - Configuration UI

### Research Directions

1. **Improved Clustering**
   - Try GPU-accelerated FAISS
   - Experiment with BERT-based models
   - Test hierarchical clustering

2. **Automated Answer Generation**
   - Train FAQ answer model
   - Integrate with LLMs
   - Validate answer quality

3. **Predictive Analytics**
   - Seasonal query forecasting
   - Emerging topic detection
   - Regional trend analysis

---

## Support & Resources

### Documentation
- [README.md](../README.md) - Project overview
- [SETUP.md](../SETUP.md) - Quick start guide
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [Dashboard README](../dashboards/README.md) - Dashboard details

### External Resources
- [Sentence Transformers Docs](https://www.sbert.net/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [React Documentation](https://react.dev/)
- [Vite Guide](https://vitejs.dev/guide/)

### Team Contacts
- **Data Science Lead**: [Name/Email]
- **Frontend Lead**: [Name/Email]
- **DevOps Lead**: [Name/Email]
- **Product Owner**: [Name/Email]

---

## Glossary

| Term | Definition |
|------|------------|
| **Clustering** | Grouping similar queries together using ML algorithms |
| **Dense Embedding** | Vector representation capturing semantic meaning (768-dim) |
| **Sparse Embedding** | TF-IDF vector capturing keyword importance |
| **Saturation Curve** | Graph showing coverage vs effort (Pareto curve) |
| **Coverage** | Percentage of queries addressed by top N question types |
| **Hinglish** | Mix of Hindi and English (common in Indian queries) |
| **Representative** | Most typical query in a cluster |
| **FAQ** | Frequently Asked Question (cluster representative) |
| **Top 100** | 100 most frequent question types (0.13% of total) |

---

## Change Log

### Version 1.0 (February 27, 2026)
- Initial team guide creation
- Complete pipeline documentation
- Dashboard component specifications
- Troubleshooting guide
- Performance benchmarks

---

**Questions?** Contact the team lead or open a GitHub Discussion.

**Found an issue?** Please report it on GitHub Issues.

**Want to contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md).

---

*This guide is a living document. Please keep it updated as the project evolves.*
