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

export default function DistributionChart({ crops }) {
    // Sort crops by query volume to show the long tail
    const sortedCrops = [...crops].sort((a, b) => b.queries - a.queries)

    // To make the chart readable, we'll show the top 15 individually
    // and then group the rest into buckets

    const top15 = sortedCrops.slice(0, 15)
    const rest = sortedCrops.slice(15)

    const labels = top15.map(c => c.name)
    const dataValues = top15.map(c => c.queries)

    // Add an "All Others" bar to show the scale of the tail
    const othersTotal = rest.reduce((sum, c) => sum + c.queries, 0)

    // For the chart, we might want to focus on the top crops to show the drop-off
    // Let's create a chart that shows Top 20 specific crops to visualize the power law

    const top20 = sortedCrops.slice(0, 20)

    const data = {
        labels: top20.map(c => c.name),
        datasets: [{
            label: 'Total Queries',
            data: top20.map(c => c.queries),
            backgroundColor: (ctx) => {
                // Highlight top 3
                return ctx.dataIndex < 3 ? '#10b981' : '#6366f1' // Emerald for top 3, Indigo for rest
            },
            borderRadius: 4,
        }]
    }

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleColor: '#fff',
                bodyColor: '#e5e7eb',
                padding: 12,
                cornerRadius: 8,
                callbacks: {
                    label: (ctx) => `${ctx.parsed.y.toLocaleString()} queries`
                }
            },
            annotation: {
                annotations: {
                    label1: {
                        type: 'label',
                        xValue: 2,
                        yValue: top20[0].queries * 0.8,
                        content: ['Vital Few', '(Top 3 drive volume)'],
                        color: '#10b981',
                        font: { size: 14 }
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: '#374151' },
                ticks: { color: '#9ca3af' },
                title: {
                    display: true,
                    text: 'Number of Queries',
                    color: '#9ca3af'
                }
            },
            x: {
                grid: { display: false },
                ticks: {
                    color: '#e5e7eb',
                    autoSkip: false,
                    maxRotation: 45,
                    minRotation: 45
                }
            }
        }
    }

    return (
        <div className="section">
            <h2 className="section-title">📊 Long Tail Analysis: Crop Distribution</h2>
            <p className="section-description">
                The dataset follows a classic "Long Tail" distribution (Power Law).
                A handful of major crops like <strong>Paddy, Wheat, and Sugarcane</strong> drive the massive volume of KCC traffic.
                <br />
                This graph shows the drop-off in query volume for the top 20 crops, illustrating where impact is concentrated.
            </p>
            <div className="chart-container" style={{ height: '500px' }}>
                <Bar data={data} options={options} />
            </div>
        </div>
    )
}
