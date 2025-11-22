"use client";

import { useState } from 'react';
import { extractFeaturesFromText } from '@/utils/nlp';
import { useAppState } from '@/context/AppState';
import type { ChatMessage, PredictionPayload } from '@/types';

export function ChatAssistant() {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [speechSupported] = useState(typeof window !== 'undefined' && 'webkitSpeechRecognition' in window);
  let recognition: any = null;
  if (speechSupported) {
    const SpeechRecognition = (window as any).webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
  }
  const { chatHistory, addChatMessage, setFormValues } = useAppState();

  const processNLPInput = (text: string): string => {
    const cleaned = text.trim();
    if (!cleaned) {
      return "I didn't catch anything. Could you repeat the readings?";
    }

    const extracted = extractFeaturesFromText(cleaned);
    console.debug('[ChatAssistant] extracted features:', extracted);

    // Update form values with extracted features
    const extractedValues: Partial<PredictionPayload> = {};
    for (const [field, value] of Object.entries(extracted)) {
      if (value !== null) {
        extractedValues[field as keyof PredictionPayload] = value;
      }
    }

    if (Object.keys(extractedValues).length > 0) {
      console.debug('[ChatAssistant] will setFormValues with:', extractedValues);
      setFormValues(extractedValues);
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
    processInput(input.trim());
  };

  const processInput = (text: string) => {
    const userMessage: ChatMessage = {
      role: 'user',
      content: text,
    };
    const response = processNLPInput(text);
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: response,
    };
    addChatMessage(userMessage);
    addChatMessage(assistantMessage);
    setInput('');
  };

  const handleSpeechInput = () => {
    if (!speechSupported || !recognition) return;
    setIsListening(true);
    recognition.start();
    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      processInput(transcript);
      setIsListening(false);
    };
    recognition.onerror = () => {
      setIsListening(false);
    };
    recognition.onend = () => {
      setIsListening(false);
    };
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
        <div
          className="chat-messages"
          style={{
            maxHeight: '320px',
            minHeight: '200px',
            overflowY: 'auto',
            paddingRight: '4px',
            marginBottom: '1rem',
            background: 'rgba(0,0,0,0.03)',
            borderRadius: '0.5rem',
          }}
        >
          {chatHistory.map((message, index) => (
            <div
              key={index}
              className={`chat-message ${message.role} ${
                message.role === 'user' ? 'user' : 'assistant'
              }`}
              style={{
                color: message.role === 'user' ? '#444' : '#fff',
                background: message.role === 'user' ? '#f6e27a' : '#222',
                borderRadius: '0.75rem',
                marginBottom: '0.5rem',
                padding: '0.75rem 1rem',
                boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
                textAlign: 'left',
                wordBreak: 'break-word',
              }}
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
          {speechSupported && (
            <button
              type="button"
              onClick={handleSpeechInput}
              className={`px-4 py-2 bg-yellow-400 hover:bg-yellow-500 text-black font-semibold rounded-md transition-colors ${isListening ? 'opacity-60' : ''}`}
              disabled={isListening}
              title="Speak your readings"
            >
              {isListening ? 'Listening…' : '🎤 Speak'}
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

