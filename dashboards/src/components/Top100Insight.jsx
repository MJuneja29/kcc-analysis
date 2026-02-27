import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
    DoughnutController
} from 'chart.js'
import { Doughnut } from 'react-chartjs-2'

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
    DoughnutController
)

export default function Top100Insight({ saturationData }) {
    if (!saturationData) return null

    const { metadata } = saturationData
    const top100Coverage = metadata.top_100_coverage || 38.62
    const totalQueries = metadata.total_queries
    const queriesCovered = Math.round(totalQueries * top100Coverage / 100)

    const data = {
        labels: ['Top 100 FAQs', 'Remaining Questions'],
        datasets: [{
            data: [top100Coverage, 100 - top100Coverage],
            backgroundColor: [
                '#10b981', // Emerald
                '#374151'  // Gray
            ],
            borderColor: ['#059669', '#1f2937'],
            borderWidth: 2
        }]
    }

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    color: '#e5e7eb',
                    font: { size: 12 },
                    padding: 15
                }
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        return `${context.label}: ${context.parsed}%`
                    }
                }
            }
        }
    }

    return (
        <section className="saturation-section">
            <div className="section-header">
                <h2>🎯 Top 100 FAQs Impact</h2>
                <p className="section-subtitle">
                    Just <strong>100 question types</strong> (0.13% of all unique questions) address{' '}
                    <strong>{top100Coverage}%</strong> of all farmer queries.
                </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'center' }}>
                <div style={{ height: '300px' }}>
                    <Doughnut data={data} options={options} />
                </div>

                <div>
                    <div className="insight-box">
                        <h3>💡 Quick Win Opportunity</h3>
                        <p>
                            By creating standardized answers for just <strong>100 FAQs</strong>, you can:
                        </p>
                        <ul style={{ marginTop: '1rem', lineHeight: '1.8' }}>
                            <li>✅ Automatically resolve <strong>{queriesCovered.toLocaleString()}</strong> queries</li>
                            <li>✅ Reduce call center workload by <strong>{top100Coverage}%</strong></li>
                            <li>✅ Provide instant responses to most common questions</li>
                            <li>✅ Free up agents for complex queries</li>
                        </ul>
                    </div>

                    <div className="stats-grid" style={{ marginTop: '1.5rem' }}>
                        <div className="stat-card-small highlight">
                            <div className="stat-label">Question Types</div>
                            <div className="stat-value">100</div>
                            <div className="stat-detail">0.13% of total</div>
                        </div>
                        <div className="stat-card-small highlight">
                            <div className="stat-label">Queries Covered</div>
                            <div className="stat-value">{(queriesCovered / 1000000).toFixed(2)}M</div>
                            <div className="stat-detail">{top100Coverage}% of total</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}
