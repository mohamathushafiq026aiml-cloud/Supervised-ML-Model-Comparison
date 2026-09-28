import React, { useState } from 'react';
import { Grid, Eye, Check, AlertTriangle } from 'lucide-react';
import { ComparisonReport, TrainedModelResult } from '../ml/types';

interface ConfusionMatrixProps {
  report: ComparisonReport;
  activeModelId?: string;
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  report,
  activeModelId
}) => {
  // Classification models only
  const classificationModels = report.models.filter(
    m => m.result.taskType === 'classification'
  );

  const bestModel = classificationModels.find(m => m.isBest) || classificationModels[0];
  const [selectedModelId, setSelectedModelId] = useState<string>(
    activeModelId || bestModel?.id || ''
  );

  const currentModel = classificationModels.find(m => m.id === selectedModelId) || bestModel;
  if (!currentModel || currentModel.result.taskType !== 'classification') {
    return null;
  }

  const { confusionMatrix, classes, accuracy, precision, recall } = currentModel.result.metrics;
  const numClasses = confusionMatrix.length;

  // Compute total samples in test
  let totalTest = 0;
  confusionMatrix.forEach(row => {
    row.forEach(val => {
      totalTest += val;
    });
  });

  // Calculate maximum cell count for heatmap color scaling
  let maxCell = 1;
  confusionMatrix.forEach(row => {
    row.forEach(val => {
      if (val > maxCell) maxCell = val;
    });
  });

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Header and Model Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Grid className="w-4 h-4 text-indigo-600" />
              <span>Confusion Matrix Heatmap</span>
            </h2>
            {currentModel.isBest && (
              <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/60">
                Best Performer
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluating true vs predicted classification on the {totalTest} held-out test samples
          </p>
        </div>

        {/* Model Switcher Dropdown */}
        <div className="flex items-center gap-2">
          <label htmlFor="cm-model-select" className="text-xs text-slate-500 font-medium">Model:</label>
          <select
            id="cm-model-select"
            value={currentModel.id}
            onChange={e => setSelectedModelId(e.target.value)}
            className="text-xs font-semibold py-1.5 px-3 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {classificationModels.map(m => (
              <option key={m.id} value={m.id}>
                {m.name} {m.isBest ? '★ (Best)' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Heatmap & Matrix Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* The Matrix */}
        <div className="md:col-span-2 flex flex-col items-center">
          {/* Top Label: Predicted */}
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
            Predicted Class →
          </div>

          <div className="flex items-center">
            {/* Left Label: True / Actual (vertical) */}
            <div className="text-xs font-bold uppercase tracking-wider text-slate-500 -rotate-90 origin-center whitespace-nowrap mr-2 select-none">
              Actual Class →
            </div>

            <div className="border border-slate-200 rounded-lg overflow-hidden bg-white shadow-xs">
              {/* Header Row: Predicted Class Names */}
              <div className="grid" style={{ gridTemplateColumns: `80px repeat(${numClasses}, minmax(100px, 1fr))` }}>
                <div className="bg-slate-50 p-2.5 text-[11px] font-semibold text-slate-400 border-b border-r border-slate-200 text-center">
                  Actual \ Pred
                </div>
                {classes.map((clsName, colIdx) => (
                  <div
                    key={colIdx}
                    className="bg-slate-50 p-2.5 text-xs font-semibold text-slate-700 border-b border-r last:border-r-0 border-slate-200 text-center truncate"
                    title={clsName}
                  >
                    {clsName}
                  </div>
                ))}
              </div>

              {/* Rows */}
              {confusionMatrix.map((row, rowIdx) => {
                const actualClass = classes[rowIdx] || `Class ${rowIdx}`;
                return (
                  <div
                    key={rowIdx}
                    className="grid border-b last:border-b-0 border-slate-200"
                    style={{ gridTemplateColumns: `80px repeat(${numClasses}, minmax(100px, 1fr))` }}
                  >
                    {/* Actual Class Label */}
                    <div className="bg-slate-50 p-3 text-xs font-semibold text-slate-700 border-r border-slate-200 flex items-center justify-center text-center truncate" title={actualClass}>
                      {actualClass}
                    </div>

                    {/* Cells */}
                    {row.map((count, colIdx) => {
                      const isCorrect = rowIdx === colIdx;
                      const intensity = Math.min(1, count / maxCell);
                      const percent = totalTest > 0 ? ((count / totalTest) * 100).toFixed(1) : '0';

                      // Determine cell semantic tag for binary classification
                      let tag = '';
                      if (numClasses === 2) {
                        if (rowIdx === 1 && colIdx === 1) tag = 'TP (True Pos)';
                        else if (rowIdx === 0 && colIdx === 0) tag = 'TN (True Neg)';
                        else if (rowIdx === 0 && colIdx === 1) tag = 'FP (Type I Error)';
                        else if (rowIdx === 1 && colIdx === 0) tag = 'FN (Type II Error)';
                      }

                      return (
                        <div
                          key={colIdx}
                          className={`p-3.5 border-r last:border-r-0 border-slate-200 flex flex-col items-center justify-center transition-colors relative group ${
                            isCorrect
                              ? 'bg-emerald-50/70 hover:bg-emerald-100/70'
                              : count > 0
                              ? 'bg-rose-50/70 hover:bg-rose-100/70'
                              : 'bg-white hover:bg-slate-50'
                          }`}
                        >
                          <span
                            className={`text-base font-bold font-mono ${
                              isCorrect ? 'text-emerald-900' : count > 0 ? 'text-rose-900' : 'text-slate-400'
                            }`}
                          >
                            {count}
                          </span>
                          <span className="text-[10px] text-slate-500 font-mono mt-0.5">
                            {percent}% of test
                          </span>
                          {tag && (
                            <span
                              className={`text-[9px] font-semibold mt-1 uppercase tracking-tight ${
                                isCorrect ? 'text-emerald-700' : count > 0 ? 'text-rose-700 font-bold' : 'text-slate-400'
                              }`}
                            >
                              {tag}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Matrix Diagnostics / Diagnostic Metrics */}
        <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Performance Breakdown
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">Model Accuracy:</span>
              <span className="font-mono font-bold text-indigo-700">
                {(accuracy * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">Precision:</span>
              <span className="font-mono font-semibold text-slate-800">
                {(precision * 100).toFixed(1)}%
              </span>
            </div>

            <div className="flex items-center justify-between pb-1 border-b border-slate-200">
              <span className="text-slate-600">Recall / Sensitivity:</span>
              <span className="font-mono font-semibold text-slate-800">
                {(recall * 100).toFixed(1)}%
              </span>
            </div>

            {numClasses === 2 && (
              <div className="pt-2 text-[11px] text-slate-500 leading-relaxed">
                <span className="font-semibold text-slate-700">Clinical Impact:</span>{' '}
                {confusionMatrix[1]?.[0] > 0
                  ? `${confusionMatrix[1][0]} false negatives occurred (high priority to minimize in medical diagnosis).`
                  : 'Zero false negatives observed on the test set!'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
