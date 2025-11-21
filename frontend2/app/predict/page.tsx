
'use client';

import { useState } from 'react';
import Link from 'next/link';
import { PredictionForm } from '@/components/PredictionForm';
import { ChatAssistant } from '@/components/ChatAssistant';
import { PredictionCard } from '@/components/PredictionCard';
import { config } from '@/config';
import type { PredictionPayload, ChatMessage } from '@/types';

export default function PredictPage() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: "Describe the latest mine readings or use the microphone. I'll fill in the numeric inputs.",
    },
  ]);
  const [formValues, setFormValues] = useState<PredictionPayload>({
    CMRR: 70.0,
    PRSUP: 40.0,
    depth_of_cover: 200.0,
    intersection_diagonal: 5.0,
    mining_height: 2.5,
  });

  const handleFormSubmit = async (values: PredictionPayload) => {
    setError(null);
    setPrediction(null);
    // Prediction will be handled by PredictionForm component
  };

  const handlePrediction = (pred: number) => {
    setPrediction(pred);
    setError(null);
  };

  const handleError = (err: string) => {
    setError(err);
    setPrediction(null);
  };

  const handleFormUpdate = (values: PredictionPayload) => {
    setFormValues(values);
  };

  const handleChatUpdate = (history: ChatMessage[]) => {
    setChatHistory(history);
  };

  const handleNLPExtraction = (values: Partial<PredictionPayload>) => {
    console.debug('[PredictPage] handleNLPExtraction received:', values);
    setFormValues((prev) => {
      const merged = { ...prev, ...values };
      console.debug('[PredictPage] formValues will be:', merged);
      return merged;
    });
  };

  return (
    <div className="mine-intel min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header / Welcome Dashboard */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            ⛏️ {config.appTitle}
          </h1>
          <p className="text-gray-600 dark:text-gray-300 mb-6">{config.appDescription}</p>

          {/* Dashboard hero with quick actions */}
          <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800/60 rounded-xl shadow-md p-6 flex flex-col md:flex-row items-center gap-4">
            <div className="flex-1 text-left">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Welcome</h2>
              <p className="text-sm text-gray-500 dark:text-gray-300">Use the quick actions below to explore predictions, important features, and data visualizations.</p>
            </div>
            <div className="flex gap-3">
              <Link href="/predict" className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-md inline-block text-center">
                Predict RFR
              </Link>
              <Link href="/features" className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-md inline-block text-center">
                Important Features
              </Link>
              <Link href="/graphs" className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-md inline-block text-center">
                Graph Analysis
              </Link>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Predict Section Anchor */}
          <div id="predict-section" className="col-span-1 lg:col-span-3 -mt-6 pt-6" />
          {/* Form Section */}
          <div className="lg:col-span-2">
            <PredictionForm
              initialValues={formValues}
              onSubmit={handleFormSubmit}
              onPrediction={handlePrediction}
              onError={handleError}
              onValuesChange={handleFormUpdate}
            />
            {/* Prediction Result */}
            {prediction !== null && (
              <div className="mt-6">
                <PredictionCard prediction={prediction} />
              </div>
            )}
            {/* Error Display */}
            {error && (
              <div className="mt-6 p-4 bg-red-100 dark:bg-red-900 border border-red-400 dark:border-red-700 text-red-700 dark:text-red-200 rounded-lg">
                <p className="font-semibold">Error:</p>
                <p>{error}</p>
              </div>
            )}
          </div>
          {/* Chat Section */}
          <div className="lg:col-span-1">
            <ChatAssistant />
          </div>
        </div>
      </div>
    </div>
  );
}
