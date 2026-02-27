# 🌾 UP Crop FAQ Dashboard - React App

An interactive React-based dashboard for analyzing frequently asked questions from 7.5 million farmer queries in Uttar Pradesh.

## 📊 Features

- **Top 10 Crops Display**: View the most queried crops with visualizations and FAQ tables
- **Distribution Chart**: Interactive Chart.js visualization showing crop query distribution
- **Crop Explorer**: Search and explore all 303 crops with detailed FAQ breakdowns
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Search**: Filter crops instantly as you type
- **Modern UI**: Clean, professional interface with smooth animations

## 🚀 Quick Start

### Development Server
```bash
npm run dev
```
Opens at: http://localhost:5174/

### Production Build
```bash
npm run build
```
Output: `dist/` folder

### Preview Production Build
```bash
npm run preview
```

## 📁 Project Structure

```
crop-faq-dashboard/
├── src/
│   ├── components/
│   │   ├── Header.jsx           # Dashboard header
│   │   ├── StatsCards.jsx       # Statistics cards
│   │   ├── TopCrops.jsx         # Top 10 crops display
│   │   ├── DistributionChart.jsx # Query distribution chart
│   │   └── CropExplorer.jsx     # Searchable crop explorer
│   ├── data/
│   │   └── cropData.json        # All crop FAQ data (303 crops)
│   ├── App.jsx                  # Main application component
│   ├── App.css                  # Application styles
│   └── main.jsx                 # Entry point
├── public/
│   └── figures/
│       └── crop_faqs/           # 303 crop visualization PNGs
└── package.json
```

## 📊 Data Overview

- **Total Queries**: 3.87 million
- **Crops**: 303
- **Question Themes**: 60
- **Data Source**: GPU-accelerated semantic clustering analysis

## 🎨 Technologies Used

- **React 18**: Modern UI framework
- **Vite**: Fast build tool and dev server
- **Chart.js**: Interactive charts
- **React-ChartJS-2**: React wrapper for Chart.js
- **CSS3**: Modern styling with gradients and animations

## 🔧 Development

### Install Dependencies
```bash
npm install
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## 📦 Building for Production

1. Build the app:
```bash
npm run build
```

2. The `dist/` folder contains the production-ready files

3. Deploy the `dist/` folder to any static hosting service:
   - Netlify
   - Vercel
   - GitHub Pages
   - AWS S3
   - Any web server

## 🌐 Deployment

### Option 1: Static Hosting
Simply upload the `dist/` folder to your hosting provider.

### Option 2: Local Server
```bash
cd dist
python3 -m http.server 8000
```
Then open: http://localhost:8000

## 📈 Key Components

### Header
Displays dashboard title and description

### StatsCards
Shows key metrics:
- Total queries analyzed
- Number of crops
- Question themes
- Analysis method (GPU)

### TopCrops
Displays top 10 crops with:
- Query count
- Visualization image
- Top 5 FAQ table

### DistributionChart
Bar chart showing crop distribution by query volume ranges

### CropExplorer
Interactive search and detailed view:
- Real-time search filtering
- Click any crop to view details
- Complete FAQ table with all themes
- Full visualization display

## 🎯 Usage

1. **View Top Crops**: Scroll to see the 10 most queried crops
2. **Analyze Distribution**: Check the chart to understand dataset composition
3. **Search Crops**: Type crop name in search box (e.g., "Wheat", "Mango")
4. **View Details**: Click any crop to see complete FAQ breakdown

## 📝 Data Format

Crop data is stored in `src/data/cropData.json`:

```json
{
  "name": "Wheat",
  "filename": "Wheat",
  "queries": 477571,
  "faqs": [
    {
      "question": "farmer asked query on weather",
      "count": 170981,
      "percentage": 17.23
    }
  ]
}
```

## 🔄 Updating Data

To update crop data:

1. Run the data generation script:
```bash
python3 ../../analysis/build_dashboard.py
```

2. Extract JSON from generated HTML:
```bash
# Extract cropData from HTML and save to src/data/cropData.json
```

3. Rebuild the app:
```bash
npm run build
```

## 🎨 Customization

### Colors
Edit `src/App.css` to change the color scheme:
- Primary gradient: `#667eea` to `#764ba2`
- Accent color: `#667eea`

### Layout
Modify grid layouts in `src/App.css`:
- `.crop-grid`: Top crops layout
- `.stats`: Statistics cards layout
- `.crop-list`: Crop explorer grid

## 📊 Performance

- **Bundle Size**: ~200KB (gzipped)
- **Load Time**: < 1 second
- **Data Size**: 0.24MB JSON
- **Images**: 303 PNGs (~287 total)

## 🐛 Troubleshooting

### Images Not Loading
- Ensure `public/figures/crop_faqs/` contains all PNG files
- Check image paths in components

### Data Not Loading
- Verify `src/data/cropData.json` exists
- Check JSON format is valid

### Build Errors
```bash
rm -rf node_modules package-lock.json
npm install
npm run build
```

## 📄 License

This project is part of the KCC Analysis system.

## 👥 Credits

- **Data Analysis**: GPU-accelerated semantic clustering
- **Dashboard**: React + Vite + Chart.js
- **Design**: Modern gradient UI with smooth animations
