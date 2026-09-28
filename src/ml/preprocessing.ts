import { Dataset, SplitData } from './types';

export function prepareData(
  dataset: Dataset,
  selectedFeatures: string[],
  targetColumn: string,
  testRatio: number = 0.2,
  randomSeed: number = 42
): SplitData {
  const rows = dataset.data;
  const isClassification = dataset.taskType === 'classification';

  // 1. Extract raw targets & encode if classification
  let targetClasses: string[] | undefined = undefined;
  let rawY: number[] = [];

  if (isClassification) {
    const rawUnique = Array.from(new Set(rows.map(r => String(r[targetColumn]))));
    // Sort unique classes (prefer 'Benign' as 0, 'Malignant' as 1 or alphabetical)
    if (rawUnique.includes('Benign') && rawUnique.includes('Malignant')) {
      targetClasses = ['Benign', 'Malignant'];
    } else {
      targetClasses = rawUnique.sort();
    }

    rawY = rows.map(r => {
      const idx = targetClasses!.indexOf(String(r[targetColumn]));
      return idx >= 0 ? idx : 0;
    });
  } else {
    rawY = rows.map(r => {
      const val = Number(r[targetColumn]);
      return isNaN(val) ? 0 : val;
    });
  }

  // 2. Extract feature matrix X
  const rawX: number[][] = rows.map(r => {
    return selectedFeatures.map(feat => {
      const v = Number(r[feat]);
      return isNaN(v) ? 0 : v;
    });
  });

  const nSamples = rows.length;
  const indices = Array.from({ length: nSamples }, (_, i) => i);

  // 3. Train/Test Split (Stratified for Classification, Shuffled for Regression)
  let trainIndices: number[] = [];
  let testIndices: number[] = [];

  // Seeded PRNG for reproducible split
  let s = randomSeed;
  const nextRandom = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };

  const shuffle = <T>(array: T[]): T[] => {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(nextRandom() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  if (isClassification && targetClasses && targetClasses.length > 1) {
    // Group indices by class for stratified split
    const classToIndices: Record<number, number[]> = {};
    for (let i = 0; i < nSamples; i++) {
      const cls = rawY[i];
      if (!classToIndices[cls]) classToIndices[cls] = [];
      classToIndices[cls].push(i);
    }

    Object.keys(classToIndices).forEach(clsKey => {
      const clsIndices = shuffle(classToIndices[Number(clsKey)]);
      const nTestForClass = Math.max(1, Math.round(clsIndices.length * testRatio));
      const testPart = clsIndices.slice(0, nTestForClass);
      const trainPart = clsIndices.slice(nTestForClass);
      testIndices.push(...testPart);
      trainIndices.push(...trainPart);
    });
  } else {
    // Regression random shuffle
    const shuffled = shuffle(indices);
    const nTest = Math.max(1, Math.round(nSamples * testRatio));
    testIndices = shuffled.slice(0, nTest);
    trainIndices = shuffled.slice(nTest);
  }

  // 4. Separate train and test sets
  const X_train_raw = trainIndices.map(i => rawX[i]);
  const y_train = trainIndices.map(i => rawY[i]);
  const X_test_raw = testIndices.map(i => rawX[i]);
  const y_test = testIndices.map(i => rawY[i]);

  const nFeatures = selectedFeatures.length;

  // 5. Standardize Numeric Features (Z-Score Standardization)
  // Fit scaler ONLY on train set: mean and standard deviation
  const scalerMean: number[] = new Array(nFeatures).fill(0);
  const scalerStd: number[] = new Array(nFeatures).fill(1);

  for (let j = 0; j < nFeatures; j++) {
    let sum = 0;
    for (let i = 0; i < X_train_raw.length; i++) {
      sum += X_train_raw[i][j];
    }
    scalerMean[j] = sum / Math.max(1, X_train_raw.length);

    let sumSqDiff = 0;
    for (let i = 0; i < X_train_raw.length; i++) {
      const diff = X_train_raw[i][j] - scalerMean[j];
      sumSqDiff += diff * diff;
    }
    const variance = sumSqDiff / Math.max(1, X_train_raw.length);
    scalerStd[j] = Math.sqrt(variance) || 1e-6; // avoid division by zero
  }

  // Transform both train and test using train scaler
  const X_train = X_train_raw.map(row =>
    row.map((val, j) => (val - scalerMean[j]) / scalerStd[j])
  );

  const X_test = X_test_raw.map(row =>
    row.map((val, j) => (val - scalerMean[j]) / scalerStd[j])
  );

  return {
    X_train,
    y_train,
    X_test,
    y_test,
    featureNames: selectedFeatures,
    targetName: targetColumn,
    targetClasses,
    scalerMean,
    scalerStd
  };
}
