'use client';

import { useState, useEffect } from 'react';
import { predictRoofFallRate } from '@/services/api-client';
import { validatePayload } from '@/utils/validators';
import { useAppState } from '@/context/AppState';
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
  depth_of_cover: {
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
  initialValues?: PredictionPayload;
  onSubmit?: (values: PredictionPayload) => void;
  onPrediction?: (prediction: number) => void;
  onError?: (error: string) => void;
  onValuesChange?: (values: PredictionPayload) => void;
}

export function PredictionForm({
  initialValues: propInitialValues,
  onSubmit,
  onPrediction,
  onError,
  onValuesChange,
}: PredictionFormProps) {
  const app = useAppState();
  const initialValues = propInitialValues ?? app.formValues;
  const [values, setValues] = useState<PredictionPayload>(initialValues);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // On mount, set local state from initialValues
    setValues(initialValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Merge context formValues (from chat) into local state, but only for fields that differ
  useEffect(() => {
    Object.entries(app.formValues).forEach(([key, value]) => {
      if (values[key as keyof PredictionPayload] !== value && value !== undefined) {
        setValues((prev) => ({ ...prev, [key]: value as number }));
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [app.formValues]);

  useEffect(() => {
    // notify parent callback if provided
    if (onValuesChange) onValuesChange(values);
  }, [values, onValuesChange]);

  const handleChange = (field: keyof PredictionPayload, value: number) => {
    setValues((prev) => {
      const updated = { ...prev, [field]: value };
      if (app) app.setFormValues(updated);
      return updated;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
  if (onError) onError('');
  if (app) app.setError(null);

    const validation = validatePayload(values);
    if (!validation.valid) {
      const msg = validation.errors.join(' • ');
      if (onError) onError(msg);
      if (app) app.setError(msg);
      setIsLoading(false);
      return;
    }

  if (onSubmit) onSubmit(values);

    try {
      const response = await predictRoofFallRate(values);
  if (onPrediction) onPrediction(response.prediction);
  if (app) app.setPrediction(response.prediction);
    } catch (err) {
  const msg = err instanceof Error ? err.message : 'An unknown error occurred';
  if (onError) onError(msg);
  if (app) app.setError(msg);
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

