import React, { useRef, useState } from 'react';
import {
  Upload,
  Database,
  Sliders,
  CheckSquare,
  Square,
  Play,
  FileSpreadsheet,
  AlertCircle,
  HelpCircle,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import { Dataset, TaskType } from '../ml/types';
import { parseCSV } from '../ml/datasets';

interface SidebarProps {
  taskType: TaskType;
  dataset: Dataset;
  onDatasetChange: (newDataset: Dataset) => void;
  selectedFeatures: string[];
  onToggleFeature: (feature: string) => void;
  onSelectAllFeatures: (selectAll: boolean) => void;
  targetColumn: string;
  onTargetColumnChange: (target: string) => void;
  isTraining: boolean;
  onRunComparison: () => void;
  hyperparams: {
    knnK: number;
    treeMaxDepth: number;
    rfTrees: number;
  };
  onHyperparamChange: (key: string, value: number) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  taskType,
  dataset,
  onDatasetChange,
  selectedFeatures,
  onToggleFeature,
  onSelectAllFeatures,
  targetColumn,
  onTargetColumnChange,
  isTraining,
  onRunComparison,
  hyperparams,
  onHyperparamChange
}) => {
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [showHyperparams, setShowHyperparams] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    const reader = new FileReader();
    reader.onload = event => {
      const text = event.target?.result as string;
      const parsed = parseCSV(text, taskType);
      if (parsed.error || !parsed.dataset) {
        setUploadError(parsed.error || 'Failed to parse CSV file.');
      } else {
        onDatasetChange(parsed.dataset);
      }
    };
    reader.onerror = () => {
      setUploadError('Failed to read file from disk.');
    };
    reader.readAsText(file);
    // Reset input so same file can be reloaded if desired
    e.target.value = '';
  };

  // Extract all columns in dataset for target selector
  const allColumns = dataset.data.length > 0 ? Object.keys(dataset.data[0]) : [];

  return (
    <aside className="w-80 border-r border-slate-200 bg-white flex flex-col h-[calc(100vh-4rem)] overflow-y-auto">
      <div className="p-5 space-y-6">
        {/* Section 1: Dataset Source */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Dataset Source
            </span>
            <span className="text-xs text-slate-400">
              {dataset.source === 'builtin' ? 'Built-in Benchmark' : 'User Upload'}
            </span>
          </div>

          {/* Current Dataset Details Card */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs space-y-2">
            <div className="flex items-center gap-2 text-slate-900 font-semibold">
              <Database className="w-4 h-4 text-indigo-600 shrink-0" />
              <span className="truncate">{dataset.name}</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              {dataset.description}
            </p>
            <div className="flex items-center gap-3 pt-1 text-[11px] text-slate-500 font-mono">
              <span>{dataset.data.length} samples</span>
              <span>·</span>
              <span>{allColumns.length} columns</span>
            </div>
          </div>

          {/* Upload CSV Dropzone */}
          <div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv,.txt"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-2.5 px-3 border border-dashed border-slate-300 rounded-lg hover:border-indigo-500 hover:bg-indigo-50/40 text-slate-600 hover:text-indigo-700 transition-all text-xs font-medium flex items-center justify-center gap-2 group"
            >
              <Upload className="w-4 h-4 text-slate-400 group-hover:text-indigo-600" />
              <span>Upload Custom CSV</span>
            </button>
            {uploadError && (
              <div className="mt-2 p-2 rounded bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-start gap-1.5">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span>{uploadError}</span>
              </div>
            )}
          </div>
        </div>

        {/* Section 2: Target Variable Selection */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label htmlFor="target-select" className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Target Variable (Y)
            </label>
            <span className="text-[11px] text-slate-400">
              {taskType === 'regression' ? 'Continuous' : 'Discrete Class'}
            </span>
          </div>

          <select
            id="target-select"
            value={targetColumn}
            onChange={e => onTargetColumnChange(e.target.value)}
            className="w-full text-xs font-medium py-2 px-3 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {allColumns.map(col => (
              <option key={col} value={col}>
                {col} {col === targetColumn ? '(Selected Target)' : ''}
              </option>
            ))}
          </select>
        </div>

        {/* Section 3: Feature Matrix Selection (X) */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Predictor Features ({selectedFeatures.length})
            </span>
            <div className="flex items-center gap-2 text-[11px]">
              <button
                type="button"
                onClick={() => onSelectAllFeatures(true)}
                className="text-indigo-600 hover:text-indigo-800 font-medium"
              >
                All
              </button>
              <span className="text-slate-300">|</span>
              <button
                type="button"
                onClick={() => onSelectAllFeatures(false)}
                className="text-slate-500 hover:text-slate-700"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="max-h-48 overflow-y-auto space-y-1 pr-1 border border-slate-200 rounded-lg p-2 bg-slate-50/50">
            {dataset.featureColumns.map(feat => {
              const isChecked = selectedFeatures.includes(feat);
              return (
                <button
                  key={feat}
                  type="button"
                  onClick={() => onToggleFeature(feat)}
                  className={`w-full text-left px-2 py-1.5 rounded text-xs flex items-center gap-2 transition-colors ${
                    isChecked
                      ? 'bg-white text-slate-900 shadow-xs border border-slate-200'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  {isChecked ? (
                    <CheckSquare className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                  ) : (
                    <Square className="w-3.5 h-3.5 text-slate-300 shrink-0" />
                  )}
                  <span className="truncate font-mono text-[11px]">{feat}</span>
                </button>
              );
            })}
          </div>
          {selectedFeatures.length === 0 && (
            <p className="text-[11px] text-amber-600 font-medium">
              Please select at least 1 feature to train models.
            </p>
          )}
        </div>

        {/* Section 4: Preprocessing Pipeline Info */}
        <div className="space-y-2 p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs">
          <div className="flex items-center gap-1.5 font-semibold text-slate-800 text-[11px]">
            <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-600" />
            <span>Preprocessing Pipeline (Automated)</span>
          </div>
          <ul className="text-[11px] text-slate-600 space-y-1 list-disc list-inside">
            <li>
              <span className="font-semibold text-slate-700">80% Train / 20% Test</span> split
            </li>
            <li>
              <span className="font-semibold text-slate-700">Z-score Standardization</span>: (x - μ) / σ
            </li>
            <li>Zero data leakage (fit on train only)</li>
          </ul>
        </div>

        {/* Section 5: Hyperparameters (Collapsible) */}
        <div className="border-t border-slate-200 pt-4">
          <button
            type="button"
            onClick={() => setShowHyperparams(!showHyperparams)}
            className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900"
          >
            <div className="flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-slate-500" />
              <span>Model Hyperparameters</span>
            </div>
            {showHyperparams ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
            )}
          </button>

          {showHyperparams && (
            <div className="mt-3 space-y-3 p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] text-slate-600 font-medium">KNN Neighbors (k)</span>
                  <span className="font-mono text-indigo-600 font-semibold">{hyperparams.knnK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="15"
                  step="2"
                  value={hyperparams.knnK}
                  onChange={e => onHyperparamChange('knnK', Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] text-slate-600 font-medium">Tree Max Depth</span>
                  <span className="font-mono text-indigo-600 font-semibold">{hyperparams.treeMaxDepth}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="8"
                  value={hyperparams.treeMaxDepth}
                  onChange={e => onHyperparamChange('treeMaxDepth', Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] text-slate-600 font-medium">Random Forest Trees</span>
                  <span className="font-mono text-indigo-600 font-semibold">{hyperparams.rfTrees}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="30"
                  step="5"
                  value={hyperparams.rfTrees}
                  onChange={e => onHyperparamChange('rfTrees', Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sticky Bottom Run Action */}
      <div className="p-4 border-t border-slate-200 mt-auto bg-white">
        <button
          type="button"
          onClick={onRunComparison}
          disabled={isTraining || selectedFeatures.length === 0}
          className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-medium text-xs rounded-lg transition-all shadow-sm flex items-center justify-center gap-2 group"
        >
          {isTraining ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Training 4 Models...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-white text-white transition-transform group-hover:scale-110" />
              <span>Train & Compare Models</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
