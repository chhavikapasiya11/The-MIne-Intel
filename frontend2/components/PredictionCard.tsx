'use client';

import { config } from '@/config';

interface PredictionCardProps {
  prediction: number;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const formattedPrediction = prediction.toFixed(config.defaultPrecision);

  return (
    <div className="bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg shadow-xl p-6 text-white">
      <h3 className="text-xl font-semibold mb-2">Predicted Roof Fall Rate</h3>
      <div className="text-4xl font-bold mb-2">{formattedPrediction}</div>
      <p className="text-primary-100 text-sm">
        This prediction is based on the CatBoost model analysis of your mining parameters.
      </p>
    </div>
  );
}

