export interface FeatureFields {
  CMRR: {
    help: string;
    minValue: number;
    maxValue: number;
    value: number;
  };
  PRSUP: {
    help: string;
    minValue: number;
    maxValue: number;
    value: number;
  };
  'depth_of_ cover': {
    help: string;
    minValue: number;
    maxValue: number;
    value: number;
  };
  intersection_diagonal: {
    help: string;
    minValue: number;
    maxValue: number;
    value: number;
  };
  mining_height: {
    help: string;
    minValue: number;
    maxValue: number;
    value: number;
  };
}

export interface PredictionPayload {
  CMRR: number;
  PRSUP: number;
  depth_of_cover: number;
  intersection_diagonal: number;
  mining_height: number;
}

export interface PredictionResponse {
  prediction: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ExtractedFeatures {
  CMRR: number | null;
  PRSUP: number | null;
  depth_of_cover: number | null;
  intersection_diagonal: number | null;
  mining_height: number | null;
}

