import { Dataset, TaskType } from './types';

// Pseudo-random seeded generator for reproducible, realistic synthetic data
function createSeededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

function normalRandom(rand: () => number, mean: number, stdDev: number): number {
  const u1 = Math.max(1e-7, rand());
  const u2 = Math.max(1e-7, rand());
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z0 * stdDev;
}

// Generate realistic Diabetes regression dataset (150 samples)
export function getDiabetesDataset(): Dataset {
  const rand = createSeededRandom(42);
  const rows: Record<string, number | string>[] = [];

  for (let i = 0; i < 150; i++) {
    const age = Math.round(normalRandom(rand, 48.5, 12.0));
    const sex = rand() > 0.5 ? 1 : 0;
    const bmi = parseFloat(normalRandom(rand, 26.4, 4.2).toFixed(1));
    const bp = Math.round(normalRandom(rand, 94.0, 13.5));
    const s1_tc = Math.round(normalRandom(rand, 189.0, 34.0)); // total cholesterol
    const s2_ldl = Math.round(normalRandom(rand, 115.0, 29.0)); // ldl
    const s3_hdl = Math.round(normalRandom(rand, 49.0, 12.5)); // hdl
    const s4_tch = parseFloat((s1_tc / Math.max(20, s3_hdl)).toFixed(2)); // tch ratio
    const s5_ltg = parseFloat(normalRandom(rand, 4.6, 0.5).toFixed(2)); // log triglycerides
    const s6_glu = Math.round(normalRandom(rand, 91.0, 11.0)); // glucose

    // Disease progression score calculation with realistic linear + interaction effects + noise
    // Standard baseline ~ 150
    const linearProgression = 
      152 +
      0.3 * (age - 48) +
      (sex === 1 ? 4 : -4) +
      5.8 * (bmi - 26) +
      1.2 * (bp - 94) +
      0.15 * (s1_tc - 180) -
      0.2 * (s2_ldl - 110) -
      2.4 * (s3_hdl - 50) +
      6.5 * (s4_tch - 3.8) +
      42.0 * (s5_ltg - 4.5) +
      0.8 * (s6_glu - 90);

    // Non-linear interaction between BMI and LTG (elevated metabolic syndrome)
    const interaction = Math.max(0, (bmi - 28) * (s5_ltg - 4.7) * 4.5);
    const noise = normalRandom(rand, 0, 22.0);
    const progression = Math.max(25, Math.min(345, Math.round(linearProgression + interaction + noise)));

    rows.push({
      age: Math.max(18, Math.min(78, age)),
      sex,
      bmi: Math.max(16, Math.min(42, bmi)),
      bp: Math.max(62, Math.min(138, bp)),
      cholesterol: Math.max(110, Math.min(300, s1_tc)),
      ldl: Math.max(60, Math.min(220, s2_ldl)),
      hdl: Math.max(20, Math.min(95, s3_hdl)),
      tch: Math.max(1.5, Math.min(8.0, s4_tch)),
      ltg: Math.max(3.2, Math.min(6.1, s5_ltg)),
      glucose: Math.max(65, Math.min(145, s6_glu)),
      progression
    });
  }

  return {
    name: 'Diabetes Progression Dataset',
    taskType: 'regression',
    description: '150 patient records measuring 10 baseline health metrics to predict quantitative 1-year disease progression.',
    targetColumn: 'progression',
    featureColumns: ['age', 'sex', 'bmi', 'bp', 'cholesterol', 'ldl', 'hdl', 'tch', 'ltg', 'glucose'],
    data: rows,
    targetDescription: 'Quantitative measure of disease progression one year after baseline (scale 25 - 345).',
    source: 'builtin'
  };
}

// Generate realistic Breast Cancer classification dataset (150 samples)
export function getBreastCancerDataset(): Dataset {
  const rand = createSeededRandom(108);
  const rows: Record<string, number | string>[] = [];

  for (let i = 0; i < 150; i++) {
    // 55% benign, 45% malignant
    const isMalignant = rand() < 0.44;

    let radius_mean: number;
    let texture_mean: number;
    let perimeter_mean: number;
    let area_mean: number;
    let smoothness_mean: number;
    let compactness_mean: number;
    let concavity_mean: number;
    let symmetry_mean: number;

    if (isMalignant) {
      radius_mean = normalRandom(rand, 17.5, 3.1);
      texture_mean = normalRandom(rand, 21.6, 3.8);
      smoothness_mean = normalRandom(rand, 0.103, 0.013);
      compactness_mean = normalRandom(rand, 0.145, 0.052);
      concavity_mean = normalRandom(rand, 0.160, 0.071);
      symmetry_mean = normalRandom(rand, 0.193, 0.027);
    } else {
      radius_mean = normalRandom(rand, 12.1, 1.8);
      texture_mean = normalRandom(rand, 17.9, 3.9);
      smoothness_mean = normalRandom(rand, 0.092, 0.012);
      compactness_mean = normalRandom(rand, 0.080, 0.034);
      concavity_mean = normalRandom(rand, 0.046, 0.042);
      symmetry_mean = normalRandom(rand, 0.174, 0.024);
    }

    // Physical dependencies
    perimeter_mean = radius_mean * 6.28 * normalRandom(rand, 1.01, 0.03);
    area_mean = Math.PI * Math.pow(radius_mean, 2) * normalRandom(rand, 1.0, 0.05);

    rows.push({
      radius_mean: parseFloat(Math.max(6.5, radius_mean).toFixed(2)),
      texture_mean: parseFloat(Math.max(9.5, texture_mean).toFixed(2)),
      perimeter_mean: parseFloat(Math.max(42.0, perimeter_mean).toFixed(1)),
      area_mean: parseFloat(Math.max(140.0, area_mean).toFixed(1)),
      smoothness_mean: parseFloat(Math.max(0.05, smoothness_mean).toFixed(4)),
      compactness_mean: parseFloat(Math.max(0.02, compactness_mean).toFixed(4)),
      concavity_mean: parseFloat(Math.max(0.0, concavity_mean).toFixed(4)),
      symmetry_mean: parseFloat(Math.max(0.10, symmetry_mean).toFixed(4)),
      diagnosis: isMalignant ? 'Malignant' : 'Benign'
    });
  }

  return {
    name: 'Breast Cancer Diagnostic Dataset',
    taskType: 'classification',
    description: '150 fine needle aspirate (FNA) cell biopsy records with 8 geometric cell nuclei characteristics predicting Benign vs Malignant status.',
    targetColumn: 'diagnosis',
    featureColumns: [
      'radius_mean',
      'texture_mean',
      'perimeter_mean',
      'area_mean',
      'smoothness_mean',
      'compactness_mean',
      'concavity_mean',
      'symmetry_mean'
    ],
    data: rows,
    targetDescription: 'Cell tumor diagnostic status: Malignant (cancerous) vs Benign (non-cancerous).',
    source: 'builtin'
  };
}

// CSV Parser Helper
export function parseCSV(csvText: string, suggestedTaskType: TaskType): {
  dataset?: Dataset;
  error?: string;
} {
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 5) {
    return { error: 'CSV file must have a header row and at least 4 rows of data.' };
  }

  // Detect delimiter: comma, semicolon, tab
  const firstLine = lines[0];
  const commaCount = (firstLine.match(/,/g) || []).length;
  const semicolonCount = (firstLine.match(/;/g) || []).length;
  const tabCount = (firstLine.match(/\t/g) || []).length;

  let delimiter = ',';
  if (semicolonCount > commaCount && semicolonCount > tabCount) delimiter = ';';
  if (tabCount > commaCount && tabCount > semicolonCount) delimiter = '\t';

  const splitLine = (text: string): string[] => {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === delimiter && !inQuotes) {
        result.push(current.trim().replace(/^"|"$/g, ''));
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current.trim().replace(/^"|"$/g, ''));
    return result;
  };

  const headers = splitLine(firstLine).map(h => h.trim());
  if (headers.length < 2) {
    return { error: 'CSV must contain at least 2 columns (1 feature and 1 target).' };
  }

  const rawRows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = splitLine(lines[i]);
    if (cells.length === headers.length) {
      const row: Record<string, string> = {};
      headers.forEach((h, idx) => {
        row[h] = cells[idx];
      });
      rawRows.push(row);
    }
  }

  if (rawRows.length < 4) {
    return { error: 'Not enough valid formatted rows found in CSV.' };
  }

  // Determine column types (numeric vs categorical)
  const columnNumericCounts: Record<string, number> = {};
  headers.forEach(h => {
    columnNumericCounts[h] = 0;
  });

  rawRows.forEach(row => {
    headers.forEach(h => {
      const val = row[h];
      if (val !== '' && !isNaN(Number(val))) {
        columnNumericCounts[h]++;
      }
    });
  });

  const numericColumns = headers.filter(h => columnNumericCounts[h] / rawRows.length > 0.8);
  const categoricalColumns = headers.filter(h => !numericColumns.includes(h));

  // Auto-detect target column:
  // 1. Common target names: target, label, y, outcome, class, diagnosis, progression, price
  const candidateNames = ['target', 'label', 'outcome', 'class', 'diagnosis', 'progression', 'y', 'price', 'status'];
  let detectedTarget = headers.find(h => candidateNames.includes(h.toLowerCase()));
  if (!detectedTarget) {
    // Default to last column
    detectedTarget = headers[headers.length - 1];
  }

  // Check if target is suitable for suggestedTaskType
  const isTargetNumeric = numericColumns.includes(detectedTarget);
  let effectiveTaskType = suggestedTaskType;
  if (!isTargetNumeric && suggestedTaskType === 'regression') {
    effectiveTaskType = 'classification';
  }

  // Pre-process rows: convert numeric values to numbers, clean strings
  const cleanedRows: Record<string, number | string>[] = [];
  rawRows.forEach(row => {
    const cleanedRow: Record<string, number | string> = {};
    headers.forEach(h => {
      const val = row[h];
      if (val === '' || val === null || val === undefined) {
        cleanedRow[h] = numericColumns.includes(h) ? 0 : 'Unknown';
      } else if (numericColumns.includes(h)) {
        const num = Number(val);
        cleanedRow[h] = isNaN(num) ? 0 : num;
      } else {
        cleanedRow[h] = val;
      }
    });
    cleanedRows.push(cleanedRow);
  });

  const featureColumns = headers.filter(h => h !== detectedTarget && numericColumns.includes(h));

  if (featureColumns.length === 0) {
    return { error: 'Could not find any numeric feature columns to train models.' };
  }

  return {
    dataset: {
      name: 'Uploaded Dataset',
      taskType: effectiveTaskType,
      description: `Uploaded dataset with ${cleanedRows.length} rows and ${headers.length} columns.`,
      targetColumn: detectedTarget,
      featureColumns,
      data: cleanedRows,
      source: 'uploaded'
    }
  };
}
