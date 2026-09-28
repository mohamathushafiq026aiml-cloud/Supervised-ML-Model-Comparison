import React from 'react';
import { Award, TrendingDown, Lightbulb, CheckCircle2 } from 'lucide-react';
import { ComparisonReport } from '../ml/types';

interface DiagnosticCardProps {
  report: ComparisonReport;
}

export const DiagnosticCard: React.FC<DiagnosticCardProps> = ({ report }) => {
  const isRegression = report.taskType === 'regression';
  const bestModel = report.models.find(m => m.isBest) || report.models[0];

  let primaryMetricDisplay = '';
  if (bestModel.result.taskType === 'regression') {
    primaryMetricDisplay = `R² = ${bestModel.result.metrics.r2.toFixed(3)}`;
  } else {
    primaryMetricDisplay = `Accuracy = ${(bestModel.result.metrics.accuracy * 100).toFixed(1)}%`;
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs space-y-4">
      {/* Top Banner: Best Performing Model Highlight */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-indigo-50/90 via-indigo-50/50 to-white border border-indigo-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-sm shrink-0">
            <Award className="w-5 h-5 text-amber-300" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700">
                Top Performing Architecture
              </span>
              <span className="text-slate-300">·</span>
              <span className="text-xs font-mono font-semibold text-indigo-900">
                {primaryMetricDisplay}
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 tracking-tight">
              {bestModel.name}
            </h3>
          </div>
        </div>

        <div className="text-xs text-slate-500 font-mono sm:text-right">
          <span>Trained in {bestModel.trainingTimeMs} ms</span>
          <span className="mx-1.5 text-slate-300">·</span>
          <span>{report.testCount} test samples</span>
        </div>
      </div>

      {/* Explanations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Why Winner Won */}
        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-emerald-800">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Why {bestModel.name} Performed Best</span>
          </div>
          <p className="text-xs text-slate-700 leading-relaxed">
            {report.explanation.winnerReason}
          </p>
        </div>

        {/* Why Lowest Lagged */}
        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200/80 space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-amber-800">
            <TrendingDown className="w-4 h-4 text-amber-600 shrink-0" />
            <span>Comparative Tradeoff & Limitations</span>
          </div>
          <p className="text-xs text-slate-700 leading-relaxed">
            {report.explanation.loserReason}
          </p>
        </div>
      </div>

      {/* Educational ML Concept Takeaway */}
      <div className="p-3 rounded-lg bg-slate-50/60 border border-slate-200 text-xs flex items-start gap-2.5 text-slate-600">
        <Lightbulb className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold text-slate-800">Student ML Concept Takeaway:</span>
          <p className="leading-relaxed">
            {report.explanation.tradeoffSummary}
          </p>
        </div>
      </div>
    </div>
  );
};
