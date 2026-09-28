import React, { useState } from 'react';
import { Target, Activity } from 'lucide-react';
import { ComparisonReport } from '../ml/types';

interface RegressionScatterProps {
  report: ComparisonReport;
  activeModelId?: string;
}

export const RegressionScatter: React.FC<RegressionScatterProps> = ({
  report,
  activeModelId
}) => {
  const regressionModels = report.models.filter(
    m => m.result.taskType === 'regression'
  );

  const bestModel = regressionModels.find(m => m.isBest) || regressionModels[0];
  const [selectedModelId, setSelectedModelId] = useState<string>(
    activeModelId || bestModel?.id || ''
  );

  const currentModel = regressionModels.find(m => m.id === selectedModelId) || bestModel;
  if (!currentModel || currentModel.result.taskType !== 'regression') {
    return null;
  }

  const actuals = currentModel.result.actuals;
  const preds = currentModel.result.predictions;
  const metrics = currentModel.result.metrics;

  // Chart bounds
  const allValues = [...actuals, ...preds];
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const padding = (maxVal - minVal) * 0.08 || 10;
  const boundMin = Math.floor(minVal - padding);
  const boundMax = Math.ceil(maxVal + padding);
  const boundRange = boundMax - boundMin || 1;

  // SVG dimensions
  const svgWidth = 520;
  const svgHeight = 280;
  const margin = { top: 20, right: 30, bottom: 40, left: 60 };
  const innerW = svgWidth - margin.left - margin.right;
  const innerH = svgHeight - margin.top - margin.bottom;

  const getX = (val: number) => margin.left + ((val - boundMin) / boundRange) * innerW;
  const getY = (val: number) => margin.top + innerH - ((val - boundMin) / boundRange) * innerH;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header and Model Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-600" />
              <span>Actual vs. Predicted Target Scatter</span>
            </h2>
            {currentModel.isBest && (
              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/60">
                Best Performer
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Points closer to the dashed 45° diagonal line reflect higher predictive accuracy
          </p>
        </div>

        {/* Model Switcher Dropdown */}
        <div className="flex items-center gap-2">
          <label htmlFor="scatter-model-select" className="text-xs text-slate-500 font-medium">Model:</label>
          <select
            id="scatter-model-select"
            value={currentModel.id}
            onChange={e => setSelectedModelId(e.target.value)}
            className="text-xs font-semibold py-1.5 px-3 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {regressionModels.map(m => (
              <option key={m.id} value={m.id}>
                {m.name} {m.isBest ? '★ (Best)' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* Scatter Plot SVG */}
        <div className="md:col-span-2 overflow-x-auto">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto max-h-72 select-none">
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map(ratio => {
              const val = boundMin + ratio * boundRange;
              const x = margin.left + ratio * innerW;
              const y = margin.top + innerH - ratio * innerH;
              return (
                <g key={ratio}>
                  {/* Vertical grid line */}
                  <line x1={x} y1={margin.top} x2={x} y2={margin.top + innerH} stroke="#f1f5f9" strokeWidth="1" />
                  {/* Horizontal grid line */}
                  <line x1={margin.left} y1={y} x2={margin.left + innerW} y2={y} stroke="#f1f5f9" strokeWidth="1" />
                  {/* X Axis tick */}
                  <text x={x} y={margin.top + innerH + 16} textAnchor="middle" className="text-[10px] fill-slate-400 font-mono">
                    {Math.round(val)}
                  </text>
                  {/* Y Axis tick */}
                  <text x={margin.left - 8} y={y + 3} textAnchor="end" className="text-[10px] fill-slate-400 font-mono">
                    {Math.round(val)}
                  </text>
                </g>
              );
            })}

            {/* Ideal 45-degree diagonal reference line y = x */}
            <line
              x1={getX(boundMin)}
              y1={getY(boundMin)}
              x2={getX(boundMax)}
              y2={getY(boundMax)}
              stroke="#6366f1"
              strokeDasharray="4 4"
              strokeWidth="1.5"
            />

            {/* Scatter points */}
            {actuals.map((actual, idx) => {
              const pred = preds[idx];
              const cx = getX(actual);
              const cy = getY(pred);
              const err = Math.abs(actual - pred);

              return (
                <g key={idx} className="group cursor-pointer">
                  <circle
                    cx={cx}
                    cy={cy}
                    r="4.5"
                    className="fill-indigo-600/80 stroke-white stroke-1 hover:fill-amber-500 hover:r-6 transition-all"
                  />
                </g>
              );
            })}

            {/* Axis Titles */}
            <text
              x={margin.left + innerW / 2}
              y={svgHeight - 8}
              textAnchor="middle"
              className="text-[11px] font-semibold fill-slate-500"
            >
              Actual Target Values (Y)
            </text>
            <text
              x={-margin.top - innerH / 2}
              y={18}
              textAnchor="middle"
              transform="rotate(-90)"
              className="text-[11px] font-semibold fill-slate-500"
            >
              Predicted Values (Ŷ)
            </text>
          </svg>
        </div>

        {/* Residuals & Metrics Card */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Residual Distribution
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">R² Coefficient:</span>
              <span className="font-mono font-bold text-indigo-700">
                {metrics.r2.toFixed(3)}
              </span>
            </div>
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">Root Mean Sq Error:</span>
              <span className="font-mono font-semibold text-slate-800">
                {metrics.rmse.toFixed(3)}
              </span>
            </div>
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">Mean Absolute Error:</span>
              <span className="font-mono font-semibold text-slate-800">
                {metrics.mae.toFixed(3)}
              </span>
            </div>
            <div className="pt-2 text-[11px] text-slate-500 leading-relaxed">
              <span className="font-semibold text-slate-700">Diagnostics:</span>{' '}
              {metrics.r2 > 0.4
                ? 'Strong linear correlation; residuals cluster tightly around the 45-degree line.'
                : 'Residual variance is elevated; check feature selection or consider non-linear ensembles.'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
