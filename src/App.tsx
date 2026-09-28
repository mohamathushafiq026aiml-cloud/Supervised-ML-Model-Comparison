import React, { useEffect, useState, useTransition } from 'react';
import {
  getDiabetesDataset,
  getBreastCancerDataset
} from './ml/datasets';
import { runModelComparison } from './ml/comparison';
import { ComparisonReport, Dataset, TaskType } from './ml/types';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { MetricsBarChart } from './components/MetricsBarChart';
import { MetricsTable } from './components/MetricsTable';
import { DiagnosticCard } from './components/DiagnosticCard';
import { ConfusionMatrix } from './components/ConfusionMatrix';
import { RegressionScatter } from './components/RegressionScatter';
import { DatasetPreviewModal } from './components/DatasetPreviewModal';
import { Layers, Database, Cpu, PieChart } from 'lucide-react';

export default function App() {
  const [taskType, setTaskType] = useState<TaskType>('regression');
  const [dataset, setDataset] = useState<Dataset>(() => getDiabetesDataset());
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(() => getDiabetesDataset().featureColumns);
  const [targetColumn, setTargetColumn] = useState<string>(() => getDiabetesDataset().targetColumn);
  
  const [hyperparams, setHyperparams] = useState({
    knnK: 5,
    treeMaxDepth: 4,
    rfTrees: 15
  });

  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [activeModelId, setActiveModelId] = useState<string | undefined>(undefined);
  const [isDataPreviewOpen, setIsDataPreviewOpen] = useState<boolean>(false);

  // Train models whenever dataset, features, or hyperparams change
  const executeComparison = (
    currentDataset: Dataset,
    features: string[],
    target: string,
    params = hyperparams
  ) => {
    if (features.length === 0) return;
    setIsTraining(true);

    // Use micro-timeout to let the UI show spinner feedback
    setTimeout(() => {
      try {
        const result = runModelComparison(currentDataset, features, target, params);
        setReport(result);
        const best = result.models.find(m => m.isBest) || result.models[0];
        setActiveModelId(best?.id);
      } catch (err) {
        console.error('Error running model comparison:', err);
      } finally {
        setIsTraining(false);
      }
    }, 60);
  };

  // Initial load
  useEffect(() => {
    executeComparison(dataset, selectedFeatures, targetColumn, hyperparams);
  }, []);

  // Switch task type
  const handleTaskTypeChange = (newTask: TaskType) => {
    if (newTask === taskType) return;
    setTaskType(newTask);

    const newDataset = newTask === 'regression' ? getDiabetesDataset() : getBreastCancerDataset();
    setDataset(newDataset);
    setSelectedFeatures(newDataset.featureColumns);
    setTargetColumn(newDataset.targetColumn);

    executeComparison(newDataset, newDataset.featureColumns, newDataset.targetColumn, hyperparams);
  };

  // Switch dataset (e.g. from CSV upload)
  const handleDatasetChange = (newDataset: Dataset) => {
    setDataset(newDataset);
    setTaskType(newDataset.taskType);
    setSelectedFeatures(newDataset.featureColumns);
    setTargetColumn(newDataset.targetColumn);

    executeComparison(newDataset, newDataset.featureColumns, newDataset.targetColumn, hyperparams);
  };

  // Reset to benchmark default
  const handleReset = () => {
    const defaultDs = taskType === 'regression' ? getDiabetesDataset() : getBreastCancerDataset();
    setDataset(defaultDs);
    setSelectedFeatures(defaultDs.featureColumns);
    setTargetColumn(defaultDs.targetColumn);
    setHyperparams({ knnK: 5, treeMaxDepth: 4, rfTrees: 15 });

    executeComparison(defaultDs, defaultDs.featureColumns, defaultDs.targetColumn, {
      knnK: 5,
      treeMaxDepth: 4,
      rfTrees: 15
    });
  };

  // Feature toggle
  const handleToggleFeature = (feat: string) => {
    let next: string[];
    if (selectedFeatures.includes(feat)) {
      next = selectedFeatures.filter(f => f !== feat);
    } else {
      next = [...selectedFeatures, feat];
    }
    setSelectedFeatures(next);
  };

  const handleSelectAllFeatures = (selectAll: boolean) => {
    if (selectAll) {
      setSelectedFeatures([...dataset.featureColumns]);
    } else {
      setSelectedFeatures([]);
    }
  };

  const handleTargetColumnChange = (newTarget: string) => {
    setTargetColumn(newTarget);
    // Remove new target from selected features if it's there
    const updatedFeatures = selectedFeatures.filter(f => f !== newTarget);
    setSelectedFeatures(updatedFeatures);
    executeComparison(dataset, updatedFeatures, newTarget, hyperparams);
  };

  const handleHyperparamChange = (key: string, value: number) => {
    const updated = { ...hyperparams, [key]: value };
    setHyperparams(updated);
  };

  return (
    <div className="min-h-screen bg-slate-100/70 text-slate-800 flex flex-col font-sans antialiased">
      {/* Top Header */}
      <Header
        taskType={taskType}
        onTaskChange={handleTaskTypeChange}
        report={report}
        onReset={handleReset}
        onOpenDataPreview={() => setIsDataPreviewOpen(true)}
      />

      {/* Main 2-Panel Layout: Left Sidebar + Right Viewport */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          taskType={taskType}
          dataset={dataset}
          onDatasetChange={handleDatasetChange}
          selectedFeatures={selectedFeatures}
          onToggleFeature={handleToggleFeature}
          onSelectAllFeatures={handleSelectAllFeatures}
          targetColumn={targetColumn}
          onTargetColumnChange={handleTargetColumnChange}
          isTraining={isTraining}
          onRunComparison={() =>
            executeComparison(dataset, selectedFeatures, targetColumn, hyperparams)
          }
          hyperparams={hyperparams}
          onHyperparamChange={handleHyperparamChange}
        />

        {/* Right Main Content */}
        <main className="flex-1 overflow-y-auto p-6 lg:p-8 space-y-6">
          {/* Top Partition & Dataset Overview Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-3.5 bg-white rounded-xl border border-slate-200 shadow-2xs space-y-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                Active Problem Task
              </span>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-900 capitalize">
                  {taskType} Learning
                </span>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">
                  4 Models
                </span>
              </div>
            </div>

            <div className="p-3.5 bg-white rounded-xl border border-slate-200 shadow-2xs space-y-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                Total Dataset Size
              </span>
              <div className="text-sm font-bold font-mono text-slate-900">
                {dataset.data.length} <span className="text-xs font-normal text-slate-500 font-sans">records</span>
              </div>
            </div>

            <div className="p-3.5 bg-white rounded-xl border border-slate-200 shadow-2xs space-y-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                Train / Test Partition
              </span>
              <div className="text-sm font-bold font-mono text-slate-900">
                80% <span className="text-xs text-slate-400 font-normal">({report ? report.trainCount : '-'})</span> / 20% <span className="text-xs text-slate-400 font-normal">({report ? report.testCount : '-'})</span>
              </div>
            </div>

            <div className="p-3.5 bg-white rounded-xl border border-slate-200 shadow-2xs space-y-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                Standardized Features (X)
              </span>
              <div className="text-sm font-bold font-mono text-indigo-700">
                {selectedFeatures.length} <span className="text-xs font-normal text-slate-500 font-sans">active columns</span>
              </div>
            </div>
          </div>

          {/* Results Sections */}
          {report ? (
            <div className="space-y-6">
              {/* 1. Best Model Highlight & Plain-English Diagnostic Explanation */}
              <DiagnosticCard report={report} />

              {/* 2. Bar Chart of Primary Performance Metric */}
              <MetricsBarChart
                report={report}
                activeModelId={activeModelId}
                onSelectModel={id => setActiveModelId(id)}
              />

              {/* 3. Comprehensive Results Table */}
              <MetricsTable
                report={report}
                activeModelId={activeModelId}
                onSelectModel={id => setActiveModelId(id)}
              />

              {/* 4. Domain Diagnostic: Confusion Matrix for Classification or Scatter Plot for Regression */}
              {taskType === 'classification' ? (
                <ConfusionMatrix report={report} activeModelId={activeModelId} />
              ) : (
                <RegressionScatter report={report} activeModelId={activeModelId} />
              )}
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-3">
              <div className="w-10 h-10 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto" />
              <p className="text-sm font-semibold text-slate-800">
                Initializing & Training Machine Learning Models...
              </p>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Partitioning dataset 80/20 and standardizing numeric features via Z-score scaling.
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Dataset Inspection Modal */}
      <DatasetPreviewModal
        isOpen={isDataPreviewOpen}
        onClose={() => setIsDataPreviewOpen(false)}
        dataset={dataset}
        targetColumn={targetColumn}
      />
    </div>
  );
}
