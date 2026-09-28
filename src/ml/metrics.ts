import { ClassificationMetrics, RegressionMetrics } from './types';

// Calculate regression metrics: MAE, RMSE, R²
export function evaluateRegression(actuals: number[], predictions: number[]): RegressionMetrics {
  const n = actuals.length;
  if (n === 0) {
    return { mae: 0, rmse: 0, r2: 0 };
  }

  let sumAbsErr = 0;
  let sumSqErr = 0;
  let sumActual = 0;

  for (let i = 0; i < n; i++) {
    const err = actuals[i] - predictions[i];
    sumAbsErr += Math.abs(err);
    sumSqErr += err * err;
    sumActual += actuals[i];
  }

  const meanActual = sumActual / n;
  let sumTot = 0;
  for (let i = 0; i < n; i++) {
    const diff = actuals[i] - meanActual;
    sumTot += diff * diff;
  }

  const mae = sumAbsErr / n;
  const rmse = Math.sqrt(sumSqErr / n);
  const r2 = sumTot > 1e-8 ? 1 - sumSqErr / sumTot : 0;

  return {
    mae: parseFloat(mae.toFixed(3)),
    rmse: parseFloat(rmse.toFixed(3)),
    r2: parseFloat(r2.toFixed(3))
  };
}

// Calculate classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
export function evaluateClassification(
  actuals: number[],
  predictions: number[],
  classLabels?: string[]
): ClassificationMetrics {
  const n = actuals.length;
  if (n === 0) {
    return {
      accuracy: 0,
      precision: 0,
      recall: 0,
      f1: 0,
      confusionMatrix: [[0, 0], [0, 0]],
      classes: classLabels || ['0', '1']
    };
  }

  // Determine all class identifiers present
  const presentClasses = Array.from(new Set([...actuals, ...predictions])).sort((a, b) => a - b);
  const numClasses = Math.max(2, Math.max(...presentClasses) + 1);

  // Initialize confusion matrix [actual][predicted]
  const cm: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  for (let i = 0; i < n; i++) {
    const act = actuals[i];
    const pred = predictions[i];
    if (cm[act] && cm[act][pred] !== undefined) {
      cm[act][pred]++;
    }
  }

  // Labels
  const classes = classLabels && classLabels.length === numClasses
    ? classLabels
    : Array.from({ length: numClasses }, (_, i) => `Class ${i}`);

  let totalCorrect = 0;
  for (let i = 0; i < numClasses; i++) {
    totalCorrect += cm[i][i];
  }
  const accuracy = totalCorrect / n;

  if (numClasses === 2) {
    // Binary classification metrics focused on positive class = 1
    const tp = cm[1][1];
    const fp = cm[0][1];
    const fn = cm[1][0];
    const tn = cm[0][0];

    const precision = tp + fp > 0 ? tp / (tp + fp) : (tp === 0 && fp === 0 ? 1 : 0);
    const recall = tp + fn > 0 ? tp / (tp + fn) : (tp === 0 && fn === 0 ? 1 : 0);
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      accuracy: parseFloat(accuracy.toFixed(3)),
      precision: parseFloat(precision.toFixed(3)),
      recall: parseFloat(recall.toFixed(3)),
      f1: parseFloat(f1.toFixed(3)),
      confusionMatrix: cm,
      classes
    };
  }

  // Multiclass: Macro-averaged Precision, Recall, F1
  let macroPrecisionSum = 0;
  let macroRecallSum = 0;
  let validClasses = 0;

  for (let c = 0; c < numClasses; c++) {
    const tp = cm[c][c];
    let fp = 0;
    let fn = 0;
    for (let row = 0; row < numClasses; row++) {
      if (row !== c) fp += cm[row][c];
    }
    for (let col = 0; col < numClasses; col++) {
      if (col !== c) fn += cm[c][col];
    }

    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;

    macroPrecisionSum += prec;
    macroRecallSum += rec;
    validClasses++;
  }

  const macroPrecision = validClasses > 0 ? macroPrecisionSum / validClasses : 0;
  const macroRecall = validClasses > 0 ? macroRecallSum / validClasses : 0;
  const macroF1 = macroPrecision + macroRecall > 0 
    ? (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall) 
    : 0;

  return {
    accuracy: parseFloat(accuracy.toFixed(3)),
    precision: parseFloat(macroPrecision.toFixed(3)),
    recall: parseFloat(macroRecall.toFixed(3)),
    f1: parseFloat(macroF1.toFixed(3)),
    confusionMatrix: cm,
    classes
  };
}
