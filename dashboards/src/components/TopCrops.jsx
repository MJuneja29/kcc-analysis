import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    BarController
} from 'chart.js'
import { Bar } from 'react-chartjs-2'

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    BarController
)

export default function TopCrops({ crops }) {
    const formatNumber = (num) => {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
        if (num >= 1000) return (num / 1000).toFixed(1) + 'k'
        return num.toString()
    }

    return (
        <div className="section">
            <h2 className="section-title">📊 Top 10 Crops by Query Volume</h2>
            <div className="crop-grid">
                {crops.map((crop) => (
                    <div key={crop.filename} className="crop-card">
                        <h3>
                            {crop.name}
                            <span className="query-count">{formatNumber(crop.queries)} queries</span>
                        </h3>

                        {/* Replaced Image with Interactive Chart */}
                        <div style={{ height: '250px', marginBottom: '1rem' }}>
                            <Bar
                                data={{
                                    labels: crop.faqs.slice(0, 5).map(f => f.question.length > 20 ? f.question.substring(0, 20) + '...' : f.question),
                                    datasets: [{
                                        label: 'Queries',
                                        data: crop.faqs.slice(0, 5).map(f => f.count),
                                        backgroundColor: '#60a5fa', // Blue 400
                                        borderRadius: 4,
                                    }]
                                }}
                                options={{
                                    indexAxis: 'y',
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    plugins: {
                                        legend: { display: false },
                                        tooltip: {
                                            callbacks: {
                                                label: (ctx) => `${ctx.parsed.x.toLocaleString()} queries`
                                            }
                                        }
                                    },
                                    scales: {
                                        x: {
                                            grid: { color: '#374151' },
                                            ticks: { color: '#9ca3af' }
                                        },
                                        y: {
                                            grid: { display: false },
                                            ticks: { color: '#e5e7eb', font: { size: 11 } }
                                        }
                                    }
                                }}
                            />
                        </div>

                        <table className="crop-table">
                            <thead>
                                <tr>
                                    <th>Question</th>
                                    <th style={{ width: '80px' }}>Count</th>
                                    <th style={{ width: '60px' }}>%</th>
                                </tr>
                            </thead>
                            <tbody>
                                {crop.faqs.slice(0, 5).map((faq, idx) => (
                                    <tr key={idx}>
                                        <td style={{ fontSize: '0.85rem' }}>{faq.question}</td>
                                        <td style={{ fontSize: '0.85rem' }}>{formatNumber(faq.count)}</td>
                                        <td style={{ fontSize: '0.85rem' }}>{faq.percentage.toFixed(1)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ))}
            </div>
        </div>
    )
}
