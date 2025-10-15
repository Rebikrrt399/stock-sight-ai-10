export interface PredictionDatum {
  date: string;
  close: number;
  [key: string]: any;
}

export type Trend = "up" | "down";

export interface PredictionResult {
  predictedDate: string;
  predictedPrice: number;
  trend: Trend;
  confidence: number; // 0-100
}

export interface PredictionPanelProps {
  ticker: string;
  currentData: PredictionDatum[];
  onPredict: () => Promise<PredictionResult>;
  isLoading: boolean;
}
