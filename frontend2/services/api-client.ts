import axios, { AxiosError } from 'axios';
import { config } from '@/config';
import type { PredictionPayload, PredictionResponse } from '@/types';

export class APIClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'APIClientError';
  }
}

export async function predictRoofFallRate(
  payload: PredictionPayload
): Promise<PredictionResponse> {
  const url = `${config.apiBaseUrl}${config.predictEndpoint}`;

  try {
    const response = await axios.post<PredictionResponse>(url, payload, {
      timeout: 10000,
    });

    if (!response.data.prediction && response.data.prediction !== 0) {
      throw new APIClientError('Backend response is missing the `prediction` field.');
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response) {
        throw new APIClientError(
          `Backend responded with ${axiosError.response.status}: ${JSON.stringify(axiosError.response.data)}`
        );
      } else if (axiosError.request) {
        throw new APIClientError('Could not reach backend API. Please check if the server is running.');
      }
    }
    throw new APIClientError(
      error instanceof Error ? error.message : 'An unknown error occurred'
    );
  }
}

