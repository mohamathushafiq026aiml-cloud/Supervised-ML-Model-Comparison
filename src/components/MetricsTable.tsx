import React from 'react';
import { Award, CheckCircle2, Clock } from 'lucide-react';
import { ComparisonReport, TrainedModelResult } from '../ml/types';

interface MetricsTableProps {
  report: ComparisonReport;
  activeModelId?: string;
  onSelectModel: (modelId: string) => void;
}

export const MetricsTable: React.FC<MetricsTableProps> = ({
  report,
  activeModelId,
  onSelectModel
}) => {
  const isRegression = report.taskType === 'regression';

  // Sort models by primary metric descending for ranking
  const sortedModels = [...report.models].sort((a, b) => {
    if (isRegression) {
      const r2A = a.result.taskType === 'regression' ? a.result.metrics.r2 : -Infinity;
      const r2B = b.result.taskType === 'regression' ? b.result.metrics.r2 : -Infinity;
      return r2B - r2A;
    } else {
      const accA = a.result.taskType === 'classification' ? a.result.metrics.accuracy : -Infinity;
      const accB = b.result.taskType === 'classification' ? b.result.metrics.accuracy : -Infinity;
      return accB - accA;
    }
  });

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
      {/* Table Header */}
      <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-slate-900 tracking-tight">
            Comprehensive Model Evaluation Metrics
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluated on held-out 20% test partition ({report.testCount} samples)
          </p>
        </div>
        <span className="text-xs text-slate-500 font-mono">
          N = 4 Models
        </span>
      </div>

      {/* Table Element */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-500 uppercase tracking-wider text-[11px]">
              <th className="py-3 px-4 font-semibold w-12 text-center">Rank</th>
              <th className="py-3 px-4 font-semibold">Model Name & Algorithm</th>
              {isRegression ? (
                <>
                  <th className="py-3 px-4 font-semibold text-right">R² Score (↑)</th>
                  <th className="py-3 px-4 font-semibold text-right">RMSE (↓)</th>
                  <th className="py-3 px-4 font-semibold text-right">MAE (↓)</th>
                </>
              ) : (
                <>
                  <th className="py-3 px-4 font-semibold text-right">Accuracy (↑)</th>
                  <th className="py-3 px-4 font-semibold text-right">Precision (↑)</th>
                  <th className="py-3 px-4 font-semibold text-right">Recall (↑)</th>
                  <th className="py-3 px-4 font-semibold text-right">F1 Score (↑)</th>
                </>
              )}
              <th className="py-3 px-4 font-semibold text-right">Train Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sortedModels.map((model, index) => {
              const isSelected = activeModelId === model.id;
              const isWinner = model.isBest;

              return (
                <tr
                  key={model.id}
                  onClick={() => onSelectModel(model.id)}
                  className={`cursor-pointer transition-colors ${
                    isWinner
                      ? 'bg-indigo-50/40 hover:bg-indigo-50/70'
                      : isSelected
                      ? 'bg-slate-50 hover:bg-slate-100/80'
                      : 'hover:bg-slate-50/60'
                  }`}
                >
                  {/* Rank Column */}
                  <td className="py-3.5 px-4 text-center">
                    {isWinner ? (
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-indigo-600 text-amber-300 font-bold text-xs shadow-xs" title="Top Ranked Model">
                        ★
                      </span>
                    ) : (
                      <span className="text-slate-400 font-mono font-medium text-xs">
                        #{index + 1}
                      </span>
                    )}
                  </td>

                  {/* Model Name */}
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${isWinner ? 'text-indigo-900' : 'text-slate-800'}`}>
                        {model.name}
                      </span>
                      {isWinner && (
                        <span className="text-[10px] uppercase font-bold tracking-wide text-indigo-700 bg-indigo-100/80 px-2 py-0.5 rounded-md">
                          Best Model
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      {model.shortDescription}
                    </div>
                  </td>

                  {/* Metrics */}
                  {isRegression && model.result.taskType === 'regression' && (
                    <>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs font-semibold text-indigo-700">
                        {model.result.metrics.r2.toFixed(3)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                        {model.result.metrics.rmse.toFixed(3)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                        {model.result.metrics.mae.toFixed(3)}
                      </td>
                    </>
                  )}

                  {!isRegression && model.result.taskType === 'classification' && (
                    <>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs font-semibold text-indigo-700">
                        {(model.result.metrics.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                        {(model.result.metrics.precision * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs text-slate-700">
                        {(model.result.metrics.recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs font-semibold text-slate-800">
                        {model.result.metrics.f1.toFixed(3)}
                      </td>
                    </>
                  )}

                  {/* Training Time */}
                  <td className="py-3.5 px-4 text-right font-mono tabular-nums text-xs text-slate-500">
                    <span className="inline-flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-400" />
                      {model.trainingTimeMs} ms
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
