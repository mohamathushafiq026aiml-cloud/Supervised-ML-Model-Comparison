import { TaskType, TrainedModelResult } from './types';

export interface ExplanationResult {
  winnerTitle: string;
  winnerReason: string;
  loserTitle: string;
  loserReason: string;
  tradeoffSummary: string;
}

export function generateExplanation(
  taskType: TaskType,
  models: TrainedModelResult[],
  sampleCount: number,
  featureCount: number
): ExplanationResult {
  const sorted = [...models].sort((a, b) => {
    if (taskType === 'regression') {
      const r2A = a.result.taskType === 'regression' ? a.result.metrics.r2 : 0;
      const r2B = b.result.taskType === 'regression' ? b.result.metrics.r2 : 0;
      return r2B - r2A;
    } else {
      const accA = a.result.taskType === 'classification' ? a.result.metrics.accuracy : 0;
      const accB = b.result.taskType === 'classification' ? b.result.metrics.accuracy : 0;
      return accB - accA;
    }
  });

  const best = sorted[0];
  const worst = sorted[sorted.length - 1];

  let winnerReason = '';
  let loserReason = '';
  let tradeoffSummary = '';

  if (taskType === 'regression') {
    const bestR2 = best.result.taskType === 'regression' ? best.result.metrics.r2 : 0;
    const worstR2 = worst.result.taskType === 'regression' ? worst.result.metrics.r2 : 0;

    // Reason for best model
    if (best.name.includes('Random Forest')) {
      winnerReason = `Random Forest Regressor achieved the top score (R² = ${bestR2}) by aggregating an ensemble of decorrelated decision trees. By averaging multiple bootstrap models, it mitigated the high variance typical of individual decision trees and captured subtle non-linear interactions across the ${featureCount} features.`;
    } else if (best.name.includes('Linear')) {
      winnerReason = `Linear Regression performed best (R² = ${bestR2}) because the underlying feature relationships with the target are predominantly linear or monotonic. With standardized predictors and ridge stabilization, it established optimal global weights without overfitting the ${sampleCount} training samples.`;
    } else if (best.name.includes('K-Nearest')) {
      winnerReason = `K-Nearest Neighbors Regressor led the comparison (R² = ${bestR2}). Because features were standardized, Euclidean distances in the ${featureCount}-dimensional space accurately grouped similar patient profiles together, producing robust local average predictions.`;
    } else {
      winnerReason = `Decision Tree Regressor achieved the highest performance (R² = ${bestR2}) by isolating distinct sub-regions and thresholds in the feature space, efficiently segmenting target progression values without imposing linear constraints.`;
    }

    // Reason for worst model
    if (worst.name.includes('Decision Tree')) {
      loserReason = `Single Decision Tree showed lower generalization (R² = ${worstR2}). With a limited sample size (${sampleCount} records), individual tree splits are sensitive to local variance and create rigid axis-aligned step functions rather than smooth continuous progression gradients.`;
    } else if (worst.name.includes('Linear')) {
      loserReason = `Linear Regression trailed (R² = ${worstR2}) due to its inability to capture non-linear interactions or quadratic thresholds without explicit polynomial feature engineering.`;
    } else if (worst.name.includes('K-Nearest')) {
      loserReason = `KNN Regressor showed reduced performance (R² = ${worstR2}), likely impacted by distance sparsity across ${featureCount} dimensions where outliers in less informative features degrade neighbor weighting.`;
    } else {
      loserReason = `Random Forest Regressor had a lower score (R² = ${worstR2}), possibly due to tree depth limits or bootstrap subsampling variance on this specific split.`;
    }

    tradeoffSummary = `Bias-Variance Tradeoff: Linear Regression imposes high structural bias (assumes linearity) but low variance, whereas Tree ensembles trade slightly higher complexity for flexibility. On datasets with both linear clinical trends and metabolic interactions, ensemble methods generally strike the best balance.`;
  } else {
    const bestAcc = best.result.taskType === 'classification' ? (best.result.metrics.accuracy * 100).toFixed(1) : '0';
    const worstAcc = worst.result.taskType === 'classification' ? (worst.result.metrics.accuracy * 100).toFixed(1) : '0';

    if (best.name.includes('Random Forest')) {
      winnerReason = `Random Forest Classifier reached the highest accuracy (${bestAcc}%) through bootstrap aggregation (bagging). Voting across multiple trees smoothed decision boundaries and prevented over-reliance on any single geometric tumor measurement.`;
    } else if (best.name.includes('Logistic')) {
      winnerReason = `Logistic Regression led with ${bestAcc}% accuracy. Standardized cell measurements created a well-behaved log-odds decision boundary, and L2 regularization prevented over-weighting correlated geometric traits like radius and perimeter.`;
    } else if (best.name.includes('K-Nearest')) {
      winnerReason = `KNN Classifier achieved top accuracy (${bestAcc}%). In standardized feature space, benign and malignant cell clusters exhibit clear spatial separation, allowing neighbor voting to classify test biopsies with high fidelity.`;
    } else {
      winnerReason = `Decision Tree Classifier scored highest (${bestAcc}% accuracy) by identifying clear hierarchical decision boundaries (e.g. critical radius and concavity thresholds) that split benign from malignant samples cleanly.`;
    }

    if (worst.name.includes('Decision Tree')) {
      loserReason = `Decision Tree Classifier underperformed (${worstAcc}% accuracy) compared to the ensemble. A single tree is prone to greedy split selection, which can overfit training noise and produce fragile decision thresholds near class boundaries.`;
    } else if (worst.name.includes('Logistic')) {
      loserReason = `Logistic Regression had lower accuracy (${worstAcc}%) because the diagnostic boundary has non-linear geometric contours that a single hyperplane cannot completely separate.`;
    } else if (worst.name.includes('K-Nearest')) {
      loserReason = `KNN Classifier lagged behind (${worstAcc}% accuracy) as uniform neighbor voting can suffer when boundary regions contain overlapping cell characteristics.`;
    } else {
      loserReason = `Random Forest Classifier had lower comparative accuracy (${worstAcc}%), likely due to sub-sampled feature combinations that missed key diagnostic indicators in certain trees.`;
    }

    tradeoffSummary = `Diagnostic Accuracy vs Interpretability: While Logistic Regression and single Decision Trees offer clear mathematical coefficients and inspectable threshold rules, Random Forest offers higher diagnostic robustness by dampening individual split errors.`;
  }

  return {
    winnerTitle: `${best.name} performed best`,
    winnerReason,
    loserTitle: `${worst.name} had lowest performance`,
    loserReason,
    tradeoffSummary
  };
}
