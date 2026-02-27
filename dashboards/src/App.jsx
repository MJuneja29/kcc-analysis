import { useState, useEffect } from 'react'
import './App.css'
import Header from './components/Header'
import StatsCards from './components/StatsCards'
import Top100Insight from './components/Top100Insight'
import TopCrops from './components/TopCrops'
import DistributionChart from './components/DistributionChart'
import SaturationChart from './components/SaturationChart'
import CropExplorer from './components/CropExplorer'
import cropDataJson from './data/cropData.json'
import saturationDataJson from './data/saturationData.json'

function App() {
  const [cropData, setCropData] = useState([])
  const [saturationData, setSaturationData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load crop data and saturation data
    setCropData(cropDataJson)
    setSaturationData(saturationDataJson)
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading crop data...</p>
      </div>
    )
  }

  const totalQueries = saturationData?.metadata?.total_queries || cropData.reduce((sum, crop) => sum + crop.queries, 0)
  const totalCrops = cropData.length
  const totalQuestionTypes = saturationData?.metadata?.total_question_types || 0
  const coverageAt10Pct = saturationData?.metadata?.coverage_at_10pct || 0

  return (
    <div className="app">
      <Header />

      <StatsCards
        totalQueries={totalQueries}
        totalCrops={totalCrops}
        totalQuestionTypes={totalQuestionTypes}
        coverageAt10Pct={coverageAt10Pct}
      />

      <div className="content">
        {/* Top 100 FAQs Impact */}
        <Top100Insight saturationData={saturationData} />

        {/* Saturation Analysis */}
        <SaturationChart saturationData={saturationData} />

        <TopCrops crops={cropData.slice(0, 10)} />

        <DistributionChart crops={cropData} />

        <CropExplorer crops={cropData} />
      </div>
    </div>
  )
}

export default App
