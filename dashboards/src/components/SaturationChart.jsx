import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LineController
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LineController
)

function SaturationChart({ saturationData }) {
  const chartRef = useRef(null)
  const chartInstance = useRef(null)

  useEffect(() => {
    if (!chartRef.current || !saturationData) return

    const ctx = chartRef.current.getContext('2d')

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const { metadata, curve } = saturationData

    // Prepare data
    const labels = curve.map(point => point.pct_question_types.toFixed(1))
    const coverageData = curve.map(point => point.pct_queries_covered)

    chartInstance.current = new ChartJS(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Efficiency Curve',
            data: coverageData,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.1,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              color: '#9ca3af',
              font: {
                size: 12
              },
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.95)',
            titleColor: '#f3f4f6',
            bodyColor: '#d1d5db',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            callbacks: {
              title: function (context) {
                return `${context[0].label}% of Question Types`
              },
              label: function (context) {
                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}% coverage`
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: '% of Unique Question Types',
              color: '#d1d5db',
              font: {
                size: 13,
                weight: '500'
              }
            },
            ticks: {
              color: '#9ca3af',
              maxTicksLimit: 11,
              callback: function (value, index) {
                // Show every 10%
                const label = this.getLabelForValue(value)
                if (parseFloat(label) % 10 === 0) {
                  return label + '%'
                }
                return ''
              }
            },
            grid: {
              color: 'rgba(55, 65, 81, 0.3)',
              drawBorder: false
            }
          },
          y: {
            title: {
              display: true,
              text: '% of Total Farmer Queries Solved',
              color: '#d1d5db',
              font: {
                size: 13,
                weight: '500'
              }
            },
            min: 0,
            max: 100,
            ticks: {
              color: '#9ca3af',
              callback: function (value) {
                return value + '%'
              }
            },
            grid: {
              color: 'rgba(55, 65, 81, 0.3)',
              drawBorder: false
            }
          }
        }
      }
    })

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [saturationData])

  if (!saturationData) {
    return <div className="chart-loading">Loading saturation data...</div>
  }

  const { metadata } = saturationData

  return (
    <section className="saturation-section">
      <div className="section-header">
        <h2>Saturation Analysis: Efficiency Curve</h2>
        <p className="section-subtitle">
          Our analysis of <strong>{metadata.total_queries.toLocaleString()}</strong> queries
          reveals an aggressive saturation. For every crop, a tiny set of "vital" questions repeats constantly.
        </p>
      </div>

      <div className="insight-box">
        <h3>The Insight: Extreme Pareto Principle</h3>
        <p>
          The curve shows the <strong>actual cumulative coverage</strong> from our clustering analysis.
          By identifying and automating answers for just the <strong>top 10%</strong> of frequent question types, we
          effectively resolve <strong>{metadata.coverage_at_10pct}%</strong> of all incoming farmer queries.
        </p>
      </div>

      <div className="chart-container" style={{ height: '500px', marginTop: '20px' }}>
        <canvas ref={chartRef}></canvas>
      </div>

      <div className="stats-grid">
        <div className="stat-card-small">
          <div className="stat-label">Total Question Types</div>
          <div className="stat-value">{metadata.total_question_types.toLocaleString()}</div>
        </div>
        <div className="stat-card-small highlight">
          <div className="stat-label">10% Coverage</div>
          <div className="stat-value">{metadata.coverage_at_10pct}%</div>
          <div className="stat-detail">of queries solved</div>
        </div>
        <div className="stat-card-small">
          <div className="stat-label">20% Coverage</div>
          <div className="stat-value">{metadata.coverage_at_20pct}%</div>
          <div className="stat-detail">of queries solved</div>
        </div>
        <div className="stat-card-small">
          <div className="stat-label">30% Coverage</div>
          <div className="stat-value">{metadata.coverage_at_30pct}%</div>
          <div className="stat-detail">of queries solved</div>
        </div>
      </div>

      <div className="insight-footer">
        <h4>💡 Strategic Insight</h4>
        <p>
          By creating standardized answers for just <strong>{Math.round(metadata.total_question_types * 0.1).toLocaleString()}</strong> question types,
          we can automatically resolve <strong>{Math.round(metadata.total_queries * metadata.coverage_at_10pct / 100).toLocaleString()}</strong> queries.
          This represents massive efficiency gains for the Kisan Call Center system.
        </p>
      </div>
    </section>
  )
}

export default SaturationChart
