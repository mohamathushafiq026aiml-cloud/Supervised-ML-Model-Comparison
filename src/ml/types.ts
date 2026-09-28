export type TaskType = 'regression' | 'classification';

export interface Dataset {
  name: string;
  taskType: TaskType;
  description: string;
  targetColumn: string;
  featureColumns: string[];
  data: Record<string, number | string>[];
  targetDescription?: string;
  source: 'builtin' | 'uploaded';
}

export interface SplitData {
  X_train: number[][];
  y_train: number[];
  X_test: number[][];
  y_test: number[];
  featureNames: string[];
  targetName: string;
  targetClasses?: string[]; // for classification: class labels (0 -> classA, 1 -> classB)
  scalerMean: number[];
  scalerStd: number[];
}

export interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: number[][]; // [actual][predicted]
  classes: string[];
}

export type ModelMetrics = 
  | { taskType: 'regression'; metrics: RegressionMetrics; predictions: number[]; actuals: number[] }
  | { taskType: 'classification'; metrics: ClassificationMetrics; predictions: number[]; actuals: number[]; probabilities?: number[][] };

export interface TrainedModelResult {
  id: string;
  name: string;
  taskType: TaskType;
  trainingTimeMs: number;
  isBest: boolean;
  result: ModelMetrics;
  shortDescription: string;
}

export interface ComparisonReport {
  timestamp: string;
  taskType: TaskType;
  datasetName: string;
  sampleCount: number;
  trainCount: number;
  testCount: number;
  featureCount: number;
  targetColumn: string;
  models: TrainedModelResult[];
  bestModelName: string;
  explanation: {
    winnerReason: string;
    loserReason: string;
    tradeoffSummary: string;
  };
}
