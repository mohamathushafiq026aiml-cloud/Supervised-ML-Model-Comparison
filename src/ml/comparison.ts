import {
  evaluateClassification,
  evaluateRegression
} from './metrics';
import {
  trainDecisionTreeClassifier,
  trainDecisionTreeRegressor,
  trainKNNClassifier,
  trainKNNRegressor,
  trainLinearRegression,
  trainLogisticRegression,
  trainRandomForestClassifier,
  trainRandomForestRegressor
} from './models';
import { prepareData } from './preprocessing';
import { generateExplanation } from './explanations';
import { ComparisonReport, Dataset, TrainedModelResult } from './types';

export interface TrainingProgressCallback {
  (currentModelName: string, completedCount: number, total: number): void;
}

export function runModelComparison(
  dataset: Dataset,
  selectedFeatures: string[],
  targetColumn: string,
  hyperparams?: {
    knnK?: number;
    treeMaxDepth?: number;
    rfTrees?: number;
  }
): ComparisonReport {
  const kVal = hyperparams?.knnK || 5;
  const maxDepthVal = hyperparams?.treeMaxDepth || 4;
  const rfTreesVal = hyperparams?.rfTrees || 15;

  const split = prepareData(dataset, selectedFeatures, targetColumn, 0.2, 42);
  const isRegression = dataset.taskType === 'regression';
  const trainedModels: TrainedModelResult[] = [];

  if (isRegression) {
    // 1. Linear Regression
    const t0 = performance.now();
    const lrPreds = trainLinearRegression(split);
    const lrTime = Math.max(1, Math.round(performance.now() - t0));
    const lrMetrics = evaluateRegression(split.y_test, lrPreds);
    trainedModels.push({
      id: 'linear_regression',
      name: 'Linear Regression',
      taskType: 'regression',
      trainingTimeMs: lrTime,
      isBest: false,
      shortDescription: 'Ordinary Least Squares with L2 regularization',
      result: {
        taskType: 'regression',
        metrics: lrMetrics,
        predictions: lrPreds,
        actuals: split.y_test
      }
    });

    // 2. Decision Tree Regressor
    const t1 = performance.now();
    const dtPreds = trainDecisionTreeRegressor(split, maxDepthVal);
    const dtTime = Math.max(1, Math.round(performance.now() - t1));
    const dtMetrics = evaluateRegression(split.y_test, dtPreds);
    trainedModels.push({
      id: 'decision_tree_regressor',
      name: 'Decision Tree Regressor',
      taskType: 'regression',
      trainingTimeMs: dtTime,
      isBest: false,
      shortDescription: `CART algorithm (max depth = ${maxDepthVal})`,
      result: {
        taskType: 'regression',
        metrics: dtMetrics,
        predictions: dtPreds,
        actuals: split.y_test
      }
    });

    // 3. Random Forest Regressor
    const t2 = performance.now();
    const rfPreds = trainRandomForestRegressor(split, rfTreesVal, maxDepthVal);
    const rfTime = Math.max(1, Math.round(performance.now() - t2));
    const rfMetrics = evaluateRegression(split.y_test, rfPreds);
    trainedModels.push({
      id: 'random_forest_regressor',
      name: 'Random Forest Regressor',
      taskType: 'regression',
      trainingTimeMs: rfTime,
      isBest: false,
      shortDescription: `Ensemble of ${rfTreesVal} bootstrap trees`,
      result: {
        taskType: 'regression',
        metrics: rfMetrics,
        predictions: rfPreds,
        actuals: split.y_test
      }
    });

    // 4. K-Nearest Neighbors Regressor
    const t3 = performance.now();
    const knnPreds = trainKNNRegressor(split, kVal);
    const knnTime = Math.max(1, Math.round(performance.now() - t3));
    const knnMetrics = evaluateRegression(split.y_test, knnPreds);
    trainedModels.push({
      id: 'knn_regressor',
      name: 'K-Nearest Neighbors Regressor',
      taskType: 'regression',
      trainingTimeMs: knnTime,
      isBest: false,
      shortDescription: `Instance-based learning with k = ${kVal}`,
      result: {
        taskType: 'regression',
        metrics: knnMetrics,
        predictions: knnPreds,
        actuals: split.y_test
      }
    });

    // Determine best: highest R²
    let bestIdx = 0;
    let maxR2 = -Infinity;
    trainedModels.forEach((m, idx) => {
      const r2 = m.result.taskType === 'regression' ? m.result.metrics.r2 : -Infinity;
      if (r2 > maxR2) {
        maxR2 = r2;
        bestIdx = idx;
      }
    });
    trainedModels[bestIdx].isBest = true;

  } else {
    // Classification Models
    // 1. Logistic Regression
    const t0 = performance.now();
    const logRes = trainLogisticRegression(split);
    const logTime = Math.max(1, Math.round(performance.now() - t0));
    const logMetrics = evaluateClassification(split.y_test, logRes.predictions, split.targetClasses);
    trainedModels.push({
      id: 'logistic_regression',
      name: 'Logistic Regression',
      taskType: 'classification',
      trainingTimeMs: logTime,
      isBest: false,
      shortDescription: 'Sigmoid link with gradient descent & L2 regularization',
      result: {
        taskType: 'classification',
        metrics: logMetrics,
        predictions: logRes.predictions,
        actuals: split.y_test,
        probabilities: logRes.probabilities
      }
    });

    // 2. Decision Tree Classifier
    const t1 = performance.now();
    const dtPreds = trainDecisionTreeClassifier(split, maxDepthVal);
    const dtTime = Math.max(1, Math.round(performance.now() - t1));
    const dtMetrics = evaluateClassification(split.y_test, dtPreds, split.targetClasses);
    trainedModels.push({
      id: 'decision_tree_classifier',
      name: 'Decision Tree Classifier',
      taskType: 'classification',
      trainingTimeMs: dtTime,
      isBest: false,
      shortDescription: `CART with Gini impurity (max depth = ${maxDepthVal})`,
      result: {
        taskType: 'classification',
        metrics: dtMetrics,
        predictions: dtPreds,
        actuals: split.y_test
      }
    });

    // 3. Random Forest Classifier
    const t2 = performance.now();
    const rfPreds = trainRandomForestClassifier(split, rfTreesVal, maxDepthVal);
    const rfTime = Math.max(1, Math.round(performance.now() - t2));
    const rfMetrics = evaluateClassification(split.y_test, rfPreds, split.targetClasses);
    trainedModels.push({
      id: 'random_forest_classifier',
      name: 'Random Forest Classifier',
      taskType: 'classification',
      trainingTimeMs: rfTime,
      isBest: false,
      shortDescription: `Ensemble of ${rfTreesVal} bagging decision trees`,
      result: {
        taskType: 'classification',
        metrics: rfMetrics,
        predictions: rfPreds,
        actuals: split.y_test
      }
    });

    // 4. K-Nearest Neighbors Classifier
    const t3 = performance.now();
    const knnPreds = trainKNNClassifier(split, kVal);
    const knnTime = Math.max(1, Math.round(performance.now() - t3));
    const knnMetrics = evaluateClassification(split.y_test, knnPreds, split.targetClasses);
    trainedModels.push({
      id: 'knn_classifier',
      name: 'K-Nearest Neighbors Classifier',
      taskType: 'classification',
      trainingTimeMs: knnTime,
      isBest: false,
      shortDescription: `Majority vote among k = ${kVal} neighbors`,
      result: {
        taskType: 'classification',
        metrics: knnMetrics,
        predictions: knnPreds,
        actuals: split.y_test
      }
    });

    // Determine best: highest Accuracy, tie-break with F1
    let bestIdx = 0;
    let maxAcc = -Infinity;
    let maxF1 = -Infinity;
    trainedModels.forEach((m, idx) => {
      if (m.result.taskType === 'classification') {
        const acc = m.result.metrics.accuracy;
        const f1 = m.result.metrics.f1;
        if (acc > maxAcc || (acc === maxAcc && f1 > maxF1)) {
          maxAcc = acc;
          maxF1 = f1;
          bestIdx = idx;
        }
      }
    });
    trainedModels[bestIdx].isBest = true;
  }

  const bestModel = trainedModels.find(m => m.isBest) || trainedModels[0];
  const explanations = generateExplanation(
    dataset.taskType,
    trainedModels,
    dataset.data.length,
    selectedFeatures.length
  );

  return {
    timestamp: new Date().toISOString(),
    taskType: dataset.taskType,
    datasetName: dataset.name,
    sampleCount: dataset.data.length,
    trainCount: split.X_train.length,
    testCount: split.X_test.length,
    featureCount: selectedFeatures.length,
    targetColumn,
    models: trainedModels,
    bestModelName: bestModel.name,
    explanation: {
      winnerReason: explanations.winnerReason,
      loserReason: explanations.loserReason,
      tradeoffSummary: explanations.tradeoffSummary
    }
  };
}
