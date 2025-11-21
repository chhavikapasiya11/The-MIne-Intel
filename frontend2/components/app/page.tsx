'use client';

import { useState } from 'react';
import { PredictionForm } from '@/components/PredictionForm';
import { ChatAssistant } from '@/components/ChatAssistant';
import { PredictionCard } from '@/components/PredictionCard';
import { config } from '@/config';
import type { PredictionPayload, ChatMessage } from '@/types';

export default function Home() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: "Describe the latest mine readings or use the microphone. I'll fill in the numeric inputs.",
    },
  ]);
  const [formValues, setFormValues] = useState<PredictionPayload>({
    CMRR: 50.0,
    PRSUP: 40.0,
    'depth_of_ cover': 200.0,
    intersection_diagonal: 5.0,
    mining_hight: 2.5,
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
    setFormValues((prev) => ({ ...prev, ...values }));
  };

  return (
    <div className="mine-intel min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            ⛏️ {config.appTitle}
          </h1>
          <p className="text-gray-600 dark:text-gray-300">{config.appDescription}</p>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
            <ChatAssistant
              chatHistory={chatHistory}
              onChatUpdate={handleChatUpdate}
              onExtraction={handleNLPExtraction}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

