import { SplitData } from './types';

// ==========================================
// Math Utilities & Solvers
// ==========================================

// Solve A * x = b via Gaussian Elimination with partial pivoting
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Create augmented matrix [A | b]
  const M: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Search for maximum pivot in column i
    let maxEl = Math.abs(M[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > maxEl) {
        maxEl = Math.abs(M[k][i]);
        maxRow = k;
      }
    }

    // Swap maximum row with current row
    const tmp = M[maxRow];
    M[maxRow] = M[i];
    M[i] = tmp;

    if (Math.abs(M[i][i]) < 1e-12) {
      M[i][i] = 1e-12; // Prevent division by zero for singular matrix
    }

    // Eliminate below
    for (let k = i + 1; k < n; k++) {
      const c = M[k][i] / M[i][i];
      for (let j = i; j <= n; j++) {
        if (i === j) {
          M[k][j] = 0;
        } else {
          M[k][j] -= c * M[i][j];
        }
      }
    }
  }

  // Back substitution
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = M[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }
  return x;
}

// Pseudo-random number generator for bootstrap
function seededRand(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Euclidean distance squared
function distSq(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

// ==========================================
// REGRESSION MODELS
// ==========================================

// 1. Linear Regression (with L2 Ridge regularization for stability)
export function trainLinearRegression(data: SplitData, lambda: number = 1e-3): number[] {
  const X = data.X_train;
  const y = data.y_train;
  const N = X.length;
  const d = X[0].length;

  // Add bias term (1.0) -> X_aug has dimension N x (d + 1)
  const X_aug = X.map(row => [1.0, ...row]);
  const p = d + 1;

  // Compute A = X^T * X + lambda * I (size p x p)
  const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < N; k++) {
        sum += X_aug[k][i] * X_aug[k][j];
      }
      A[i][j] = sum;
    }
    // Regularize weights (skip bias i=0 or small regularize)
    A[i][i] += (i === 0 ? 1e-6 : lambda);
  }

  // Compute b = X^T * y (size p)
  const bVec: number[] = new Array(p).fill(0);
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (let k = 0; k < N; k++) {
      sum += X_aug[k][i] * y[k];
    }
    bVec[i] = sum;
  }

  // Solve for w = (X^T X + lambda I)^(-1) X^T y
  const w = solveLinearSystem(A, bVec);

  // Predict on test data
  return data.X_test.map(row => {
    let pred = w[0]; // bias
    for (let j = 0; j < d; j++) {
      pred += w[j + 1] * row[j];
    }
    return pred;
  });
}

// 2. Decision Tree Regressor (CART)
interface RegTreeNode {
  isLeaf: boolean;
  value?: number;
  featureIndex?: number;
  threshold?: number;
  left?: RegTreeNode;
  right?: RegTreeNode;
}

function buildRegTree(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  maxDepth: number = 4,
  minSamplesSplit: number = 5,
  featureSubsetRatio: number = 1.0,
  rand?: () => number
): RegTreeNode {
  const n = indices.length;
  // Calculate node mean
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += y[indices[i]];
  }
  const meanVal = sum / Math.max(1, n);

  if (depth >= maxDepth || n <= minSamplesSplit) {
    return { isLeaf: true, value: meanVal };
  }

  // Calculate current node variance / MSE
  let currentVar = 0;
  for (let i = 0; i < n; i++) {
    const diff = y[indices[i]] - meanVal;
    currentVar += diff * diff;
  }

  const d = X[0].length;
  let featuresToConsider = Array.from({ length: d }, (_, i) => i);
  if (featureSubsetRatio < 1.0 && rand) {
    // Subsample features
    const count = Math.max(1, Math.floor(d * featureSubsetRatio));
    const shuffled = [...featuresToConsider].sort(() => rand() - 0.5);
    featuresToConsider = shuffled.slice(0, count);
  }

  let bestGain = 0;
  let bestFeat = -1;
  let bestThresh = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const feat of featuresToConsider) {
    // Get unique sorted values for split candidates
    const values = indices.map(idx => X[idx][feat]);
    values.sort((a, b) => a - b);

    // Sample candidate thresholds
    const step = Math.max(1, Math.floor(values.length / 10));
    for (let i = 0; i < values.length - 1; i += step) {
      if (values[i] === values[i + 1]) continue;
      const thresh = (values[i] + values[i + 1]) / 2;

      const leftIdx: number[] = [];
      const rightIdx: number[] = [];
      let sumL = 0;
      let sumR = 0;

      for (let j = 0; j < n; j++) {
        const rowIdx = indices[j];
        if (X[rowIdx][feat] <= thresh) {
          leftIdx.push(rowIdx);
          sumL += y[rowIdx];
        } else {
          rightIdx.push(rowIdx);
          sumR += y[rowIdx];
        }
      }

      if (leftIdx.length === 0 || rightIdx.length === 0) continue;

      const meanL = sumL / leftIdx.length;
      const meanR = sumR / rightIdx.length;

      let varL = 0;
      for (const idx of leftIdx) {
        const diff = y[idx] - meanL;
        varL += diff * diff;
      }
      let varR = 0;
      for (const idx of rightIdx) {
        const diff = y[idx] - meanR;
        varR += diff * diff;
      }

      const gain = currentVar - (varL + varR);
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = feat;
        bestThresh = thresh;
        bestLeft = leftIdx;
        bestRight = rightIdx;
      }
    }
  }

  if (bestGain <= 1e-7 || bestLeft.length === 0 || bestRight.length === 0) {
    return { isLeaf: true, value: meanVal };
  }

  return {
    isLeaf: false,
    featureIndex: bestFeat,
    threshold: bestThresh,
    left: buildRegTree(X, y, bestLeft, depth + 1, maxDepth, minSamplesSplit, featureSubsetRatio, rand),
    right: buildRegTree(X, y, bestRight, depth + 1, maxDepth, minSamplesSplit, featureSubsetRatio, rand)
  };
}

function predictRegTree(tree: RegTreeNode, x: number[]): number {
  if (tree.isLeaf || tree.featureIndex === undefined || tree.threshold === undefined) {
    return tree.value ?? 0;
  }
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictRegTree(tree.left!, x);
  }
  return predictRegTree(tree.right!, x);
}

export function trainDecisionTreeRegressor(data: SplitData, maxDepth: number = 4): number[] {
  const allIndices = Array.from({ length: data.X_train.length }, (_, i) => i);
  const tree = buildRegTree(data.X_train, data.y_train, allIndices, 0, maxDepth, 4);
  return data.X_test.map(row => predictRegTree(tree, row));
}

// 3. Random Forest Regressor
export function trainRandomForestRegressor(
  data: SplitData,
  nTrees: number = 15,
  maxDepth: number = 4
): number[] {
  const N = data.X_train.length;
  const trees: RegTreeNode[] = [];
  const rand = seededRand(1337);

  for (let t = 0; t < nTrees; t++) {
    // Bootstrap sampling
    const sampleIndices: number[] = [];
    for (let i = 0; i < N; i++) {
      sampleIndices.push(Math.floor(rand() * N));
    }
    const tree = buildRegTree(
      data.X_train,
      data.y_train,
      sampleIndices,
      0,
      maxDepth,
      3,
      0.65, // feature subset ratio
      rand
    );
    trees.push(tree);
  }

  // Predict: average over all trees
  return data.X_test.map(row => {
    let sum = 0;
    for (const tree of trees) {
      sum += predictRegTree(tree, row);
    }
    return sum / trees.length;
  });
}

// 4. K-Nearest Neighbors Regressor
export function trainKNNRegressor(data: SplitData, k: number = 5): number[] {
  const effectiveK = Math.min(k, data.X_train.length);

  return data.X_test.map(testRow => {
    const distances = data.X_train.map((trainRow, idx) => ({
      dist: distSq(testRow, trainRow),
      target: data.y_train[idx]
    }));

    // Sort by distance ascending
    distances.sort((a, b) => a.dist - b.dist);
    const nearest = distances.slice(0, effectiveK);

    let sum = 0;
    nearest.forEach(item => {
      sum += item.target;
    });
    return sum / effectiveK;
  });
}

// ==========================================
// CLASSIFICATION MODELS
// ==========================================

// 1. Logistic Regression (One-vs-Rest for multiclass support, Sigmoid for binary)
export function trainLogisticRegression(
  data: SplitData,
  epochs: number = 120,
  lr: number = 0.08,
  l2: number = 0.01
): { predictions: number[]; probabilities: number[][] } {
  const X = data.X_train;
  const y = data.y_train;
  const N = X.length;
  const d = X[0].length;

  const uniqueClasses = Array.from(new Set(y)).sort();
  const numClasses = uniqueClasses.length;

  if (numClasses <= 2) {
    // Binary Logistic Regression
    let w = new Array(d).fill(0);
    let b = 0;

    for (let ep = 0; ep < epochs; ep++) {
      let gradW = new Array(d).fill(0);
      let gradB = 0;

      for (let i = 0; i < N; i++) {
        let z = b;
        for (let j = 0; j < d; j++) {
          z += w[j] * X[i][j];
        }
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
        const err = p - (y[i] === 1 ? 1 : 0);

        for (let j = 0; j < d; j++) {
          gradW[j] += err * X[i][j];
        }
        gradB += err;
      }

      for (let j = 0; j < d; j++) {
        w[j] -= lr * (gradW[j] / N + l2 * w[j]);
      }
      b -= lr * (gradB / N);
    }

    const testProbs: number[][] = [];
    const testPreds: number[] = [];

    data.X_test.forEach(row => {
      let z = b;
      for (let j = 0; j < d; j++) {
        z += w[j] * row[j];
      }
      const p1 = 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
      const p0 = 1.0 - p1;
      testProbs.push([p0, p1]);
      testPreds.push(p1 >= 0.5 ? 1 : 0);
    });

    return { predictions: testPreds, probabilities: testProbs };
  } else {
    // One-vs-Rest for multiclass
    const classWeights: { w: number[]; b: number }[] = [];

    for (const c of uniqueClasses) {
      let w = new Array(d).fill(0);
      let b = 0;

      for (let ep = 0; ep < epochs; ep++) {
        let gradW = new Array(d).fill(0);
        let gradB = 0;

        for (let i = 0; i < N; i++) {
          let z = b;
          for (let j = 0; j < d; j++) {
            z += w[j] * X[i][j];
          }
          const p = 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
          const target = y[i] === c ? 1 : 0;
          const err = p - target;

          for (let j = 0; j < d; j++) {
            gradW[j] += err * X[i][j];
          }
          gradB += err;
        }

        for (let j = 0; j < d; j++) {
          w[j] -= lr * (gradW[j] / N + l2 * w[j]);
        }
        b -= lr * (gradB / N);
      }
      classWeights.push({ w, b });
    }

    const testProbs: number[][] = [];
    const testPreds: number[] = [];

    data.X_test.forEach(row => {
      const scores = classWeights.map(({ w, b }) => {
        let z = b;
        for (let j = 0; j < d; j++) {
          z += w[j] * row[j];
        }
        return 1.0 / (1.0 + Math.exp(-Math.max(-25, Math.min(25, z))));
      });

      const sumScores = scores.reduce((acc, v) => acc + v, 0) || 1;
      const normalizedProbs = scores.map(s => s / sumScores);
      testProbs.push(normalizedProbs);

      let maxScore = -1;
      let bestClassIdx = 0;
      scores.forEach((s, idx) => {
        if (s > maxScore) {
          maxScore = s;
          bestClassIdx = idx;
        }
      });
      testPreds.push(uniqueClasses[bestClassIdx]);
    });

    return { predictions: testPreds, probabilities: testProbs };
  }
}

// 2. Decision Tree Classifier (CART with Gini Impurity)
interface ClfTreeNode {
  isLeaf: boolean;
  predictedClass?: number;
  classDistribution?: number[];
  featureIndex?: number;
  threshold?: number;
  left?: ClfTreeNode;
  right?: ClfTreeNode;
}

function calculateGini(y: number[], indices: number[], numClasses: number): number {
  if (indices.length === 0) return 0;
  const counts = new Array(numClasses).fill(0);
  for (const idx of indices) {
    counts[y[idx]]++;
  }
  let sumSq = 0;
  const total = indices.length;
  for (let c = 0; c < numClasses; c++) {
    const p = counts[c] / total;
    sumSq += p * p;
  }
  return 1.0 - sumSq;
}

function getMajorityClass(y: number[], indices: number[], numClasses: number): number {
  const counts = new Array(numClasses).fill(0);
  for (const idx of indices) {
    counts[y[idx]]++;
  }
  let maxCount = -1;
  let majority = 0;
  for (let c = 0; c < numClasses; c++) {
    if (counts[c] > maxCount) {
      maxCount = counts[c];
      majority = c;
    }
  }
  return majority;
}

function buildClfTree(
  X: number[][],
  y: number[],
  indices: number[],
  numClasses: number,
  depth: number,
  maxDepth: number = 4,
  minSamplesSplit: number = 4,
  featureSubsetRatio: number = 1.0,
  rand?: () => number
): ClfTreeNode {
  const n = indices.length;
  const majorityClass = getMajorityClass(y, indices, numClasses);

  if (depth >= maxDepth || n <= minSamplesSplit) {
    return { isLeaf: true, predictedClass: majorityClass };
  }

  const currentGini = calculateGini(y, indices, numClasses);
  if (currentGini === 0) {
    return { isLeaf: true, predictedClass: majorityClass };
  }

  const d = X[0].length;
  let featuresToConsider = Array.from({ length: d }, (_, i) => i);
  if (featureSubsetRatio < 1.0 && rand) {
    const count = Math.max(1, Math.floor(d * featureSubsetRatio));
    const shuffled = [...featuresToConsider].sort(() => rand() - 0.5);
    featuresToConsider = shuffled.slice(0, count);
  }

  let bestGain = 0;
  let bestFeat = -1;
  let bestThresh = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const feat of featuresToConsider) {
    const values = indices.map(idx => X[idx][feat]);
    values.sort((a, b) => a - b);

    const step = Math.max(1, Math.floor(values.length / 10));
    for (let i = 0; i < values.length - 1; i += step) {
      if (values[i] === values[i + 1]) continue;
      const thresh = (values[i] + values[i + 1]) / 2;

      const leftIdx: number[] = [];
      const rightIdx: number[] = [];

      for (let j = 0; j < n; j++) {
        const rowIdx = indices[j];
        if (X[rowIdx][feat] <= thresh) {
          leftIdx.push(rowIdx);
        } else {
          rightIdx.push(rowIdx);
        }
      }

      if (leftIdx.length === 0 || rightIdx.length === 0) continue;

      const giniL = calculateGini(y, leftIdx, numClasses);
      const giniR = calculateGini(y, rightIdx, numClasses);
      const weightedGini = (leftIdx.length / n) * giniL + (rightIdx.length / n) * giniR;
      const gain = currentGini - weightedGini;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = feat;
        bestThresh = thresh;
        bestLeft = leftIdx;
        bestRight = rightIdx;
      }
    }
  }

  if (bestGain <= 1e-6 || bestLeft.length === 0 || bestRight.length === 0) {
    return { isLeaf: true, predictedClass: majorityClass };
  }

  return {
    isLeaf: false,
    featureIndex: bestFeat,
    threshold: bestThresh,
    left: buildClfTree(X, y, bestLeft, numClasses, depth + 1, maxDepth, minSamplesSplit, featureSubsetRatio, rand),
    right: buildClfTree(X, y, bestRight, numClasses, depth + 1, maxDepth, minSamplesSplit, featureSubsetRatio, rand)
  };
}

function predictClfTree(tree: ClfTreeNode, x: number[]): number {
  if (tree.isLeaf || tree.featureIndex === undefined || tree.threshold === undefined) {
    return tree.predictedClass ?? 0;
  }
  if (x[tree.featureIndex] <= tree.threshold) {
    return predictClfTree(tree.left!, x);
  }
  return predictClfTree(tree.right!, x);
}

export function trainDecisionTreeClassifier(data: SplitData, maxDepth: number = 4): number[] {
  const allIndices = Array.from({ length: data.X_train.length }, (_, i) => i);
  const numClasses = Math.max(...data.y_train) + 1;
  const tree = buildClfTree(data.X_train, data.y_train, allIndices, numClasses, 0, maxDepth, 4);
  return data.X_test.map(row => predictClfTree(tree, row));
}

// 3. Random Forest Classifier
export function trainRandomForestClassifier(
  data: SplitData,
  nTrees: number = 15,
  maxDepth: number = 4
): number[] {
  const N = data.X_train.length;
  const numClasses = Math.max(...data.y_train) + 1;
  const trees: ClfTreeNode[] = [];
  const rand = seededRand(2026);

  for (let t = 0; t < nTrees; t++) {
    const sampleIndices: number[] = [];
    for (let i = 0; i < N; i++) {
      sampleIndices.push(Math.floor(rand() * N));
    }
    const tree = buildClfTree(
      data.X_train,
      data.y_train,
      sampleIndices,
      numClasses,
      0,
      maxDepth,
      3,
      0.65,
      rand
    );
    trees.push(tree);
  }

  return data.X_test.map(row => {
    const votes = new Array(numClasses).fill(0);
    trees.forEach(tree => {
      const pred = predictClfTree(tree, row);
      votes[pred]++;
    });

    let maxVotes = -1;
    let chosenClass = 0;
    votes.forEach((v, c) => {
      if (v > maxVotes) {
        maxVotes = v;
        chosenClass = c;
      }
    });
    return chosenClass;
  });
}

// 4. K-Nearest Neighbors Classifier
export function trainKNNClassifier(data: SplitData, k: number = 5): number[] {
  const effectiveK = Math.min(k, data.X_train.length);
  const numClasses = Math.max(...data.y_train) + 1;

  return data.X_test.map(testRow => {
    const distances = data.X_train.map((trainRow, idx) => ({
      dist: distSq(testRow, trainRow),
      target: data.y_train[idx]
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const nearest = distances.slice(0, effectiveK);

    const votes = new Array(numClasses).fill(0);
    nearest.forEach(n => {
      votes[n.target]++;
    });

    let maxVotes = -1;
    let chosenClass = 0;
    votes.forEach((v, c) => {
      if (v > maxVotes) {
        maxVotes = v;
        chosenClass = c;
      }
    });
    return chosenClass;
  });
}
