import React from 'react';
import { Download, FileText, RefreshCw, BarChart2 } from 'lucide-react';
import { ComparisonReport, TaskType } from '../ml/types';
import { generateMarkdownReport, generateResultsCSV, triggerDownload } from '../ml/export';

interface HeaderProps {
  taskType: TaskType;
  onTaskChange: (task: TaskType) => void;
  report: ComparisonReport | null;
  onReset: () => void;
  onOpenDataPreview: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  taskType,
  onTaskChange,
  report,
  onReset,
  onOpenDataPreview
}) => {
  const handleExportCSV = () => {
    if (!report) return;
    const csv = generateResultsCSV(report);
    triggerDownload(csv, `${report.datasetName.toLowerCase().replace(/\s+/g, '_')}_metrics.csv`, 'text/csv');
  };

  const handleExportReport = () => {
    if (!report) return;
    const md = generateMarkdownReport(report);
    triggerDownload(md, `${report.datasetName.toLowerCase().replace(/\s+/g, '_')}_summary_report.md`, 'text/markdown');
  };

  return (
    <header className="h-16 border-b border-slate-200 bg-white px-6 flex items-center justify-between sticky top-0 z-20">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-sm">
          <BarChart2 className="w-4 h-4" />
        </div>
        <span className="text-base font-bold text-slate-900 tracking-tight whitespace-nowrap">
          Supervised ML Model Comparison
        </span>
      </div>

      {/* Zone 2: Task Mode Segmented Control */}
      <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200">
        <button
          type="button"
          onClick={() => onTaskChange('regression')}
          className={`px-3.5 py-1.5 text-xs font-semibold rounded-md transition-all whitespace-nowrap ${
            taskType === 'regression'
              ? 'bg-white text-indigo-700 shadow-sm border border-slate-200/60'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Regression Models
        </button>
        <button
          type="button"
          onClick={() => onTaskChange('classification')}
          className={`px-3.5 py-1.5 text-xs font-semibold rounded-md transition-all whitespace-nowrap ${
            taskType === 'classification'
              ? 'bg-white text-indigo-700 shadow-sm border border-slate-200/60'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Classification Models
        </button>
      </div>

      {/* Zone 3: Export & Action Buttons */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenDataPreview}
          className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap flex items-center gap-1.5"
          title="View raw dataset table and column distributions"
        >
          <span>Data Preview</span>
        </button>

        <button
          type="button"
          onClick={handleExportCSV}
          disabled={!report}
          className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap flex items-center gap-1.5"
          title="Download model evaluation table as CSV"
        >
          <Download className="w-3.5 h-3.5 text-slate-500" />
          <span>Export CSV</span>
        </button>

        <button
          type="button"
          onClick={handleExportReport}
          disabled={!report}
          className="px-3 py-1.5 text-xs font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap flex items-center gap-1.5 shadow-sm"
          title="Download comprehensive text/markdown report"
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Download Report</span>
        </button>

        <button
          type="button"
          onClick={onReset}
          className="p-1.5 text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition-colors"
          title="Reset to default benchmark dataset"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
