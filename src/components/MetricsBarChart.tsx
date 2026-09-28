import React, { useState } from 'react';
import { Award, Info } from 'lucide-react';
import { ComparisonReport, TaskType } from '../ml/types';

interface MetricsBarChartProps {
  report: ComparisonReport;
  activeModelId?: string;
  onSelectModel?: (modelId: string) => void;
}

export const MetricsBarChart: React.FC<MetricsBarChartProps> = ({
  report,
  activeModelId,
  onSelectModel
}) => {
  const isRegression = report.taskType === 'regression';

  // Available metric options to switch between
  const metricOptions = isRegression
    ? [
        { key: 'r2', label: 'R² Score (Primary)', higherIsBetter: true },
        { key: 'rmse', label: 'RMSE (Root Mean Sq Error)', higherIsBetter: false },
        { key: 'mae', label: 'MAE (Mean Absolute Error)', higherIsBetter: false }
      ]
    : [
        { key: 'accuracy', label: 'Accuracy (Primary)', higherIsBetter: true },
        { key: 'f1', label: 'F1 Score', higherIsBetter: true },
        { key: 'precision', label: 'Precision', higherIsBetter: true },
        { key: 'recall', label: 'Recall', higherIsBetter: true }
      ];

  const [selectedMetricKey, setSelectedMetricKey] = useState<string>(
    isRegression ? 'r2' : 'accuracy'
  );

  const currentOption = metricOptions.find(o => o.key === selectedMetricKey) || metricOptions[0];

  // Extract model values
  const modelEntries = report.models.map(m => {
    let val = 0;
    if (m.result.taskType === 'regression') {
      const reg = m.result.metrics;
      if (selectedMetricKey === 'r2') val = reg.r2;
      else if (selectedMetricKey === 'rmse') val = reg.rmse;
      else if (selectedMetricKey === 'mae') val = reg.mae;
    } else {
      const clf = m.result.metrics;
      if (selectedMetricKey === 'accuracy') val = clf.accuracy;
      else if (selectedMetricKey === 'f1') val = clf.f1;
      else if (selectedMetricKey === 'precision') val = clf.precision;
      else if (selectedMetricKey === 'recall') val = clf.recall;
    }

    return {
      id: m.id,
      name: m.name,
      value: val,
      isBest: m.isBest,
      description: m.shortDescription
    };
  });

  // Calculate chart scale
  const values = modelEntries.map(e => e.value);
  const minVal = Math.min(0, ...values);
  const maxVal = Math.max(0.1, ...values);
  const range = maxVal - minVal || 1;

  // Chart layout dimensions
  const svgWidth = 640;
  const svgHeight = 240;
  const margin = { top: 30, right: 30, bottom: 50, left: 160 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;
  const barHeight = 28;
  const gap = (innerHeight - modelEntries.length * barHeight) / (modelEntries.length + 1);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header & Metric Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <span>Model Performance Comparison</span>
            <span className="text-xs font-normal text-slate-500">
              ({currentOption.label})
            </span>
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            {currentOption.higherIsBetter ? 'Higher score indicates better performance' : 'Lower error indicates better fit'} on held-out 20% test data
          </p>
        </div>

        {/* Metric Selector Tabs */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg text-xs self-start sm:self-auto border border-slate-200">
          {metricOptions.map(option => (
            <button
              key={option.key}
              type="button"
              onClick={() => setSelectedMetricKey(option.key)}
              className={`px-2.5 py-1 rounded-md font-medium transition-all whitespace-nowrap ${
                selectedMetricKey === option.key
                  ? 'bg-white text-indigo-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {option.key.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* SVG Bar Chart */}
      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-auto max-h-64 font-sans select-none"
        >
          {/* Grid lines and X-axis ticks */}
          {[0, 0.25, 0.5, 0.75, 1.0].map(ratio => {
            const val = minVal + ratio * range;
            const x = margin.left + ratio * innerWidth;
            return (
              <g key={ratio}>
                <line
                  x1={x}
                  y1={margin.top}
                  x2={x}
                  y2={margin.top + innerHeight}
                  stroke="#f1f5f9"
                  strokeWidth="1"
                />
                <text
                  x={x}
                  y={margin.top + innerHeight + 18}
                  textAnchor="middle"
                  className="text-[10px] fill-slate-400 font-mono"
                >
                  {isRegression && (selectedMetricKey === 'rmse' || selectedMetricKey === 'mae')
                    ? val.toFixed(1)
                    : (val * (selectedMetricKey === 'accuracy' ? 100 : 1)).toFixed(selectedMetricKey === 'accuracy' ? 0 : 2) +
                      (selectedMetricKey === 'accuracy' ? '%' : '')}
                </text>
              </g>
            );
          })}

          {/* Zero reference line */}
          {minVal < 0 && (
            <line
              x1={margin.left + (Math.abs(minVal) / range) * innerWidth}
              y1={margin.top}
              x2={margin.left + (Math.abs(minVal) / range) * innerWidth}
              y2={margin.top + innerHeight}
              stroke="#cbd5e1"
              strokeDasharray="3 3"
              strokeWidth="1.5"
            />
          )}

          {/* Model Bars */}
          {modelEntries.map((entry, idx) => {
            const y = margin.top + gap + idx * (barHeight + gap);
            const normalizedValue = Math.max(0, (entry.value - minVal) / range);
            const barWidth = Math.max(4, normalizedValue * innerWidth);
            const isSelected = activeModelId === entry.id;
            const isWinner = entry.isBest;

            return (
              <g
                key={entry.id}
                className="cursor-pointer transition-opacity hover:opacity-95"
                onClick={() => onSelectModel && onSelectModel(entry.id)}
              >
                {/* Model Label (Left) */}
                <text
                  x={margin.left - 12}
                  y={y + barHeight / 2 + 4}
                  textAnchor="end"
                  className={`text-xs ${
                    isWinner ? 'font-bold fill-indigo-900' : 'font-medium fill-slate-700'
                  }`}
                >
                  {entry.name}
                </text>

                {/* Background track */}
                <rect
                  x={margin.left}
                  y={y}
                  width={innerWidth}
                  height={barHeight}
                  rx="6"
                  fill="#f8fafc"
                />

                {/* Primary Metric Bar */}
                <rect
                  x={margin.left}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  rx="6"
                  fill={isWinner ? '#4f46e5' : isSelected ? '#3b82f6' : '#94a3b8'}
                  className="transition-all duration-300"
                />

                {/* Winner Crown / Star Indicator */}
                {isWinner && (
                  <circle
                    cx={margin.left + barWidth - 14}
                    cy={y + barHeight / 2}
                    r="8"
                    fill="#312e81"
                  />
                )}
                {isWinner && (
                  <text
                    x={margin.left + barWidth - 14}
                    y={y + barHeight / 2 + 3}
                    textAnchor="middle"
                    fill="#fbbf24"
                    className="text-[10px] font-bold"
                  >
                    ★
                  </text>
                )}

                {/* Bar Value Label */}
                <text
                  x={margin.left + barWidth + 8}
                  y={y + barHeight / 2 + 4}
                  className={`text-xs font-mono font-semibold ${
                    isWinner ? 'fill-indigo-700' : 'fill-slate-700'
                  }`}
                >
                  {selectedMetricKey === 'accuracy'
                    ? `${(entry.value * 100).toFixed(1)}%`
                    : entry.value.toFixed(3)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Chart Footer Note */}
      <div className="flex items-center justify-between text-xs text-slate-500 pt-1 border-t border-slate-100">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-xs bg-indigo-600 inline-block" />
          <span className="font-medium text-slate-700">Top-Performing Model</span>
          <span className="text-slate-300">·</span>
          <span className="w-3 h-3 rounded-xs bg-slate-400 inline-block" />
          <span>Comparative Models</span>
        </div>
        <span className="text-[11px] text-slate-400">Click any bar to inspect model details below</span>
      </div>
    </div>
  );
};
