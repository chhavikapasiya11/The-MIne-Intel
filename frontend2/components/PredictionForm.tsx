'use client';

import { useState, useEffect } from 'react';
import { predictRoofFallRate } from '@/services/api-client';
import { validatePayload } from '@/utils/validators';
import type { PredictionPayload } from '@/types';

const FEATURE_FIELDS = {
  CMRR: {
    help: 'Coal Mine Roof Rating (0-100)',
    minValue: 0.0,
    maxValue: 100.0,
  },
  PRSUP: {
    help: 'Percentage of roof support load (0-100)',
    minValue: 0.0,
    maxValue: 100.0,
  },
  'depth_of_ cover': {
    help: 'Depth of cover in meters',
    minValue: 0.0,
    maxValue: 1000.0,
  },
  intersection_diagonal: {
    help: 'Intersection diagonal in meters',
    minValue: 0.0,
    maxValue: 20.0,
  },
  mining_hight: {
    help: 'Mining height in meters',
    minValue: 0.0,
    maxValue: 10.0,
  },
} as const;

interface PredictionFormProps {
  initialValues: PredictionPayload;
  onSubmit: (values: PredictionPayload) => void;
  onPrediction: (prediction: number) => void;
  onError: (error: string) => void;
  onValuesChange: (values: PredictionPayload) => void;
}

export function PredictionForm({
  initialValues,
  onSubmit,
  onPrediction,
  onError,
  onValuesChange,
}: PredictionFormProps) {
  const [values, setValues] = useState<PredictionPayload>(initialValues);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setValues(initialValues);
  }, [initialValues]);

  useEffect(() => {
    onValuesChange(values);
  }, [values, onValuesChange]);

  const handleChange = (field: keyof PredictionPayload, value: number) => {
    setValues((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    onError('');

    const validation = validatePayload(values);
    if (!validation.valid) {
      onError(validation.errors.join(' • '));
      setIsLoading(false);
      return;
    }

    onSubmit(values);

    try {
      const response = await predictRoofFallRate(values);
      onPrediction(response.prediction);
    } catch (err) {
      onError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
        Mining Parameters
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(FEATURE_FIELDS).map(([field, options]) => {
            const fieldKey = field as keyof PredictionPayload;
            return (
              <div key={field} className="space-y-1">
                <label
                  htmlFor={field}
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  {field.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                </label>
                <input
                  type="number"
                  id={field}
                  min={options.minValue}
                  max={options.maxValue}
                  step="0.1"
                  value={values[fieldKey]}
                  onChange={(e) => handleChange(fieldKey, parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
                  title={options.help}
                />
                <p className="text-xs text-gray-500 dark:text-gray-400">{options.help}</p>
              </div>
            );
          })}
        </div>
        <button
          type="submit"
          disabled={isLoading}
          className="w-full mt-6 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-lg shadow-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Predicting...' : 'Predict roof fall rate'}
        </button>
      </form>
    </div>
  );
}

