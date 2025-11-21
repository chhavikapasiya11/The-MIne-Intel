"use client";

import React, { createContext, useContext, useState, useCallback } from 'react';
import type { PredictionPayload, ChatMessage } from '@/types';

interface AppState {
  formValues: PredictionPayload;
  setFormValues: (v: Partial<PredictionPayload>) => void;
  prediction: number | null;
  setPrediction: (p: number | null) => void;
  error: string | null;
  setError: (e: string | null) => void;
  chatHistory: ChatMessage[];
  addChatMessage: (m: ChatMessage) => void;
}

const defaultValues: PredictionPayload = {
  CMRR: 50.0,
  PRSUP: 40.0,
  depth_of_cover: 200.0,
  intersection_diagonal: 5.0,
  mining_height: 2.5,
};

const AppStateContext = createContext<AppState | undefined>(undefined);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [formValues, setFormValuesState] = useState<PredictionPayload>(defaultValues);
  const [prediction, setPredictionState] = useState<number | null>(null);
  const [error, setErrorState] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([
    { role: 'assistant', content: "Describe the latest mine readings or use the microphone. I'll fill in the numeric inputs." },
  ]);

  const setFormValues = useCallback((v: Partial<PredictionPayload>) => {
    setFormValuesState((prev) => ({ ...prev, ...v }));
  }, []);

  const setPrediction = useCallback((p: number | null) => {
    setPredictionState(p);
    if (p !== null) setErrorState(null);
  }, []);

  const setError = useCallback((e: string | null) => {
    setErrorState(e);
    if (e) setPredictionState(null);
  }, []);

  const addChatMessage = useCallback((m: ChatMessage) => {
    setChatHistory((prev) => {
      const combined = [...prev, m].slice(-12);
      return combined;
    });
  }, []);

  const ctx: AppState = {
    formValues,
    setFormValues,
    prediction,
    setPrediction,
    error,
    setError,
    chatHistory,
    addChatMessage,
  };

  return <AppStateContext.Provider value={ctx}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error('useAppState must be used within AppStateProvider');
  return ctx;
}
