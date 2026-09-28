import React from 'react';
import { X, Table, CheckCircle2, Database } from 'lucide-react';
import { Dataset } from '../ml/types';

interface DatasetPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset: Dataset;
  targetColumn: string;
}

export const DatasetPreviewModal: React.FC<DatasetPreviewModalProps> = ({
  isOpen,
  onClose,
  dataset,
  targetColumn
}) => {
  if (!isOpen) return null;

  const rows = dataset.data.slice(0, 12);
  const columns = dataset.data.length > 0 ? Object.keys(dataset.data[0]) : [];

  // Summary stats for numeric columns
  const stats: Record<string, { min: number; max: number; avg: number }> = {};
  columns.forEach(col => {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let numericCount = 0;

    dataset.data.forEach(row => {
      const val = Number(row[col]);
      if (!isNaN(val)) {
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
        numericCount++;
      }
    });

    if (numericCount > 0) {
      stats[col] = {
        min: parseFloat(min.toFixed(2)),
        max: parseFloat(max.toFixed(2)),
        avg: parseFloat((sum / numericCount).toFixed(2))
      };
    }
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4">
      <div className="bg-white rounded-xl shadow-xl border border-slate-200 w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-700">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900 tracking-tight">
                Dataset Inspection: {dataset.name}
              </h2>
              <p className="text-xs text-slate-500">
                {dataset.data.length} total samples · {columns.length} columns · Target: <span className="font-semibold text-indigo-700 font-mono">{targetColumn}</span>
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6">
          {/* Data Sample Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Sample Data Records (First 12 Rows)
              </span>
              <span className="text-[11px] text-slate-400">
                Showing {Math.min(12, dataset.data.length)} of {dataset.data.length}
              </span>
            </div>

            <div className="border border-slate-200 rounded-lg overflow-x-auto max-h-64 shadow-xs">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 uppercase tracking-wider text-[10px]">
                    <th className="py-2 px-3 font-semibold w-10 text-center">#</th>
                    {columns.map(col => (
                      <th
                        key={col}
                        className={`py-2 px-3 font-semibold whitespace-nowrap ${
                          col === targetColumn ? 'bg-indigo-50/80 text-indigo-900 font-bold' : ''
                        }`}
                      >
                        {col}
                        {col === targetColumn && ' (Target)'}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-[11px]">
                  {rows.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="py-1.5 px-3 text-center text-slate-400 font-sans text-xs">
                        {idx + 1}
                      </td>
                      {columns.map(col => (
                        <td
                          key={col}
                          className={`py-1.5 px-3 whitespace-nowrap ${
                            col === targetColumn ? 'bg-indigo-50/30 font-semibold text-indigo-800' : 'text-slate-700'
                          }`}
                        >
                          {String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Statistical Summaries */}
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block mb-2">
              Numeric Feature Distribution Summary
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(stats).slice(0, 8).map(([col, s]) => (
                <div key={col} className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs space-y-1">
                  <div className="font-semibold text-slate-800 truncate font-mono text-[11px]" title={col}>
                    {col}
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>Min / Max:</span>
                    <span className="font-mono text-slate-700">{s.min} / {s.max}</span>
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>Mean:</span>
                    <span className="font-mono font-medium text-indigo-700">{s.avg}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors"
          >
            Close Preview
          </button>
        </div>
      </div>
    </div>
  );
};
