import type { PredictionPayload } from '@/types';

export interface ValidationError {
  field: string;
  message: string;
}

export function validatePayload(payload: PredictionPayload): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  const fieldRanges: Record<keyof PredictionPayload, { min: number; max: number; name: string }> = {
    CMRR: { min: 0, max: 100, name: 'CMRR' },
    PRSUP: { min: 0, max: 100, name: 'PRSUP' },
    depth_of_cover: { min: 0, max: 1000, name: 'Depth of cover' },
    intersection_diagonal: { min: 0, max: 20, name: 'Intersection diagonal' },
    mining_height: { min: 0, max: 10, name: 'Mining height' },
  };

  for (const [field, value] of Object.entries(payload)) {
    const fieldKey = field as keyof PredictionPayload;
    const range = fieldRanges[fieldKey];
    
    if (value === null || value === undefined || isNaN(Number(value))) {
      errors.push(`${range.name} must be a valid number`);
    } else {
      const numValue = Number(value);
      if (numValue < range.min || numValue > range.max) {
        errors.push(`${range.name} must be between ${range.min} and ${range.max}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

