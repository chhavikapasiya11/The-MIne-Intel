import type { ExtractedFeatures } from '@/types';

const FIELD_SYNONYMS: Record<keyof ExtractedFeatures, string[]> = {

  CMRR: [
    'cmrr',
    'cmr',
    'cr',
    'roof rating',
    'mass rate',
    'rock rate',
    'rating',
    'r rating',
    'geo rate',
    'stability',
    'cmrr value'
  ],

  PRSUP: [
    'prsup',
    'support load',
    'roof support',
    'ps',
    'support',
    'load',
    'roof load',
    'r load',
    'sup load',
    'press',
    'pressure',
    'p support',
    'r support'
  ],

  'depth_of_ cover': [
    'depth of cover',
    'overburden',
    'cover depth',
    'depth',
    'cover',
    'burden',
    'ob',
    'd cover',
    'deep',
    'top depth',
    'surface depth',
    'dc',
    'cd'
  ],

  intersection_diagonal: [
    'intersection diagonal',
    'diagonal',
    'crosscut diagonal',
    'diag',
    'id',
    'int diag',
    'cross diag',
    'x diag',
    'inter diag',
    'd length',
    'junction d',
    'crosscut d'
  ],

  mining_hight: [
    'mining height',
    'seam height',
    'extraction height',
    'mh',
    'height',
    'h',
    'mine h',
    'seam h',
    'ex h',
    'work h',
    'coal h',
    'm height',
    'face h'
  ],
};


function extractNumber(snippet: string): number | null {
  const match = snippet.match(/(-?\d+(?:\.\d+)?)/);
  if (!match) {
    return null;
  }
  try {
    const value = parseFloat(match[1]);
    return isNaN(value) ? null : value;
  } catch {
    return null;
  }
}

export function extractFeaturesFromText(text: string): ExtractedFeatures {
  const lowered = text.toLowerCase();
  const extracted: ExtractedFeatures = {
    CMRR: null,
    PRSUP: null,
    'depth_of_ cover': null,
    intersection_diagonal: null,
    mining_hight: null,
  };

  for (const [field, synonyms] of Object.entries(FIELD_SYNONYMS)) {
    const fieldKey = field as keyof ExtractedFeatures;
    
    for (const synonym of synonyms) {
      const escaped = synonym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const pattern = new RegExp(
        `${escaped}[^0-9-]*(-?\\d+(?:\\.\\d+)?)(?:\\s?(?:meters|meter|m|percent|%))?`,
        'i'
      );
      const match = lowered.match(pattern);
      if (match) {
        extracted[fieldKey] = parseFloat(match[1]);
        break;
      }
    }
    
    if (extracted[fieldKey] !== null) {
      continue;
    }

    // Support "value for synonym" phrasing, e.g., "set CMRR to 45"
    for (const synonym of synonyms) {
      const escaped = synonym.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const altPattern = new RegExp(
        `(?:set|make|is|at)\\s+${escaped}[^0-9-]*(-?\\d+(?:\\.\\d+)?)`,
        'i'
      );
      const match = lowered.match(altPattern);
      if (match) {
        extracted[fieldKey] = parseFloat(match[1]);
        break;
      }
    }
  }

  return extracted;
}

