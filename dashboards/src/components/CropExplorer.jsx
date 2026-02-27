import { useState } from 'react'
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
)

export default function CropExplorer({ crops }) {
    const [searchQuery, setSearchQuery] = useState('')
    const [selectedCrop, setSelectedCrop] = useState(null)

    const formatNumber = (num) => {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
        if (num >= 1000) return (num / 1000).toFixed(1) + 'k'
        return num.toString()
    }

    const filteredCrops = crops.filter(crop =>
        crop.name.toLowerCase().includes(searchQuery.toLowerCase())
    )

    const handleCropClick = (crop) => {
        setSelectedCrop(crop)
        setTimeout(() => {
            document.getElementById('cropDetails')?.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            })
        }, 100)
    }

    return (
        <div className="section">
            <h2 className="section-title">🔍 Explore Any Crop</h2>
            <div className="search-section">
                <div className="search-box">
                    <input
                        type="text"
                        placeholder="Search for a crop (e.g., Wheat, Paddy, Mango)..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
                <div className="crop-list">
                    {filteredCrops.map((crop) => (
                        <div
                            key={crop.filename}
                            className="crop-item"
                            onClick={() => handleCropClick(crop)}
                        >
                            {crop.name}
                        </div>
                    ))}
                </div>
            </div>

            {selectedCrop && (
                <div id="cropDetails" className="crop-details active">
                    <h3>
                        {selectedCrop.name}
                        <span className="query-count-large">
                            ({formatNumber(selectedCrop.queries)} total queries)
                        </span>
                    </h3>

                    {/* Interactive Chart for Selected Crop */}
                    <div style={{ height: '400px', marginBottom: '2rem' }}>
                        <Bar
                            data={{
                                labels: selectedCrop.faqs.slice(0, 10).map(f => f.question.length > 40 ? f.question.substring(0, 40) + '...' : f.question),
                                datasets: [{
                                    label: 'Queries',
                                    data: selectedCrop.faqs.slice(0, 10).map(f => f.count),
                                    backgroundColor: '#8b5cf6', // Violet 500
                                    borderRadius: 4,
                                }]
                            }}
                            options={{
                                indexAxis: 'y',
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: { display: false },
                                    title: {
                                        display: true,
                                        text: `Top 10 Frequently Asked Questions - ${selectedCrop.name}`,
                                        color: '#e5e7eb',
                                        font: { size: 16 }
                                    },
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
                                        ticks: { color: '#e5e7eb', font: { size: 12 } }
                                    }
                                }
                            }}
                        />
                    </div>

                    <table className="crop-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Question Theme</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {selectedCrop.faqs.map((faq, idx) => (
                                <tr key={idx}>
                                    <td>{idx + 1}</td>
                                    <td>{faq.question}</td>
                                    <td>{formatNumber(faq.count)}</td>
                                    <td>{faq.percentage.toFixed(2)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    )
}
