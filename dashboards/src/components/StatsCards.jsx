export default function StatsCards({ totalQueries, totalCrops, totalQuestionTypes, coverageAt10Pct }) {
    const formatNumber = (num) => {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
        if (num >= 1000) return (num / 1000).toFixed(1) + 'k'
        return num.toString()
    }

    return (
        <div className="stats">
            <div className="stat-card">
                <div className="number">{formatNumber(totalQueries)}</div>
                <div className="label">Total Queries Analyzed</div>
            </div>
            <div className="stat-card">
                <div className="number">{totalCrops}</div>
                <div className="label">Crops Analyzed</div>
            </div>
            <div className="stat-card">
                <div className="number">{formatNumber(totalQuestionTypes)}</div>
                <div className="label">Unique Question Types</div>
            </div>
            <div className="stat-card">
                <div className="number">{coverageAt10Pct}%</div>
                <div className="label">Coverage by Top 10%</div>
            </div>
        </div>
    )
}
