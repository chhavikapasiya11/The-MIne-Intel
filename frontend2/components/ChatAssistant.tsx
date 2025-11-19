'use client';

import { useState } from 'react';
import { extractFeaturesFromText } from '@/utils/nlp';
import type { ChatMessage, PredictionPayload } from '@/types';

interface ChatAssistantProps {
  chatHistory: ChatMessage[];
  onChatUpdate: (history: ChatMessage[]) => void;
  onExtraction: (values: Partial<PredictionPayload>) => void;
}

export function ChatAssistant({
  chatHistory,
  onChatUpdate,
  onExtraction,
}: ChatAssistantProps) {
  const [input, setInput] = useState('');

  const processNLPInput = (text: string): string => {
    const cleaned = text.trim();
    if (!cleaned) {
      return "I didn't catch anything. Could you repeat the readings?";
    }

    const extracted = extractFeaturesFromText(cleaned);
    
    // Update form values with extracted features
    const extractedValues: Partial<PredictionPayload> = {};
    for (const [field, value] of Object.entries(extracted)) {
      if (value !== null) {
        extractedValues[field as keyof PredictionPayload] = value;
      }
    }
    
    if (Object.keys(extractedValues).length > 0) {
      onExtraction(extractedValues);
    }

    const filled = Object.entries(extracted)
      .filter(([, value]) => value !== null)
      .map(([field, value]) => `${field}: ${value}`);
    
    const missing = Object.entries(extracted)
      .filter(([, value]) => value === null)
      .map(([field]) => field);

    let response = '';
    if (filled.length > 0) {
      response = 'Updated ' + filled.join(', ') + '.';
    } else {
      response = "I couldn't detect any numbers in that message.";
    }

    if (missing.length > 0) {
      response += ' Please provide ' + missing.join(', ') + '.';
    }

    return response;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
    };

    const newHistory = [...chatHistory, userMessage];
    onChatUpdate(newHistory);

    const response = processNLPInput(input);
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: response,
    };

    onChatUpdate([...newHistory, assistantMessage]);
    setInput('');

    // Limit chat history to 12 messages
    if (newHistory.length + 1 > 12) {
      const limited = [...newHistory, assistantMessage].slice(-12);
      onChatUpdate(limited);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
        Voice & Chat Assistant
      </h2>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Speak or type natural language. I'll populate the form.
      </p>

      <div className="chat-panel">
        <div className="chat-messages">
          {chatHistory.map((message, index) => (
            <div
              key={index}
              className={`chat-message ${message.role} ${
                message.role === 'user' ? 'user' : 'assistant'
              }`}
            >
              <span className="mr-2">
                {message.role === 'user' ? '👷' : '🤖'}
              </span>
              {message.content}
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe the readings"
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-gray-700 dark:text-white"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-semibold rounded-md transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

