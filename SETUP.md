# Quick Setup Guide

Get the KCC Analysis project running in under 5 minutes!

## Prerequisites

✅ **Python 3.8+** - Check: `python --version`  
✅ **Node.js 16+** - Check: `node --version`  
✅ **Git** - Check: `git --version`

## Step 1: Clone Repository

```bash
git clone <repository-url>
cd "KCC Analysis"
```

## Step 2: Python Setup (2 mins)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Dashboard Setup (2 mins)

```bash
cd dashboards
npm install
npm run dev
```

🎉 **Dashboard opens at:** http://localhost:5174/

## Step 4: Get Data Files (Contact Maintainer)

The repository doesn't include large data files. You need:

### Required for Analysis:
- `data/processed/kcc_master_dataset_remapped.csv` (~50MB)

### Required for Dashboard:
- `dashboards/src/data/cropData.json` (~200KB)
- `dashboards/src/data/saturationData.json` (~17KB)

**Option 1:** Contact maintainer for files  
**Option 2:** Generate from scratch (requires raw data + GPU)

## Quick Test

### Test Dashboard (without data):
```bash
cd dashboards
npm run dev
```
Opens at http://localhost:5174/ (will show loading state without data)

### Test Analysis Script:
```bash
# From project root
python analysis/saturation_analysis.py
```
(Requires clustering results in `outputs/leiden_clustering/`)

## Common Issues

### Issue: `ModuleNotFoundError`
**Solution:** Make sure virtual environment is activated and requirements installed
```bash
pip install -r requirements.txt
```

### Issue: Dashboard shows "Loading..." forever
**Solution:** Generate or obtain data files for `dashboards/src/data/`

### Issue: Analysis script fails with path errors
**Solution:** Run from project root, not from `analysis/` directory

### Issue: npm install fails
**Solution:** Update Node.js to version 16 or higher
```bash
node --version  # Should be >= 16
```

## Folder Structure

```
KCC Analysis/
├── analysis/           # Python analysis scripts
├── data/
│   ├── aggregated/    # Small summary files (in git)
│   └── processed/     # Large datasets (git-ignored)
├── outputs/
│   ├── leiden_clustering/  # Clustering results (git-ignored)
│   └── figures/           # Visualizations (git-ignored)
├── dashboards/        # React web app
└── assets/           # Images and logos
```

## Next Steps

1. **Read Documentation:**
   - [README.md](README.md) - Full project overview
   - [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) - **Complete technical guide (pipeline & dashboard architecture)**
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
   - [dashboards/README.md](dashboards/README.md) - Dashboard details

2. **Explore the Code:**
   - Check `analysis/` for Python scripts
   - Look at `dashboards/src/` for React components

3. **Run Analysis:**
   - Get data files
   - Try running the analysis scripts
   - Generate visualizations

4. **Customize Dashboard:**
   - Modify components in `dashboards/src/components/`
   - Update styles in `dashboards/src/App.css`
   - Add new features

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Test changes
python analysis/script.py  # For Python changes
npm run dev                # For dashboard changes

# 4. Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature

# 5. Create Pull Request on GitHub
```

## Getting Help

- 📖 Read [README.md](README.md) for detailed information
- 🤝 Read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- 🐛 Open an issue for bugs
- 💬 Start a discussion for questions

---

**Happy Coding!** 🌾
