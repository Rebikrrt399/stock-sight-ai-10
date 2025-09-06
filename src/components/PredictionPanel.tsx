import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { TrendingUp, TrendingDown, Zap, Target } from "lucide-react";

interface PredictionResult {
  predictedPrice: number;
  trend: 'up' | 'down';
  confidence: number;
  predictedDate: string;
}

interface PredictionPanelProps {
  ticker: string;
  currentData: any[];
  onPredict: () => Promise<PredictionResult>;
  isLoading?: boolean;
}

const PredictionPanel = ({ ticker, currentData, onPredict, isLoading }: PredictionPanelProps) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const result = await onPredict();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
    setPredicting(false);
  };

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;

  // Create chart data with prediction
  const chartData = currentData.length > 0 ? [
    ...currentData.slice(-30), // Last 30 days
    ...(prediction ? [{
      date: prediction.predictedDate,
      close: prediction.predictedPrice,
      isPredicted: true
    }] : [])
  ] : [];

  return (
    <div className="space-y-6">
      <Card className="bg-gradient-card shadow-card border-border">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-primary" />
            <CardTitle className="text-foreground">AI Price Prediction</CardTitle>
          </div>
          <CardDescription className="text-muted-foreground">
            Get AI-powered price predictions for {ticker}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button 
            onClick={handlePredict}
            disabled={currentData.length === 0 || predicting || isLoading}
            className="w-full bg-gradient-primary hover:opacity-90 text-primary-foreground shadow-primary"
          >
            {predicting ? "Analyzing..." : "Predict Next Price"}
          </Button>

          {prediction && (
            <div className="space-y-4 animate-slide-up">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="bg-background/50 border-border">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Target className="h-4 w-4 text-primary" />
                      <div>
                        <p className="text-sm text-muted-foreground">Predicted Price</p>
                        <p className="text-xl font-bold text-foreground">
                          {formatCurrency(prediction.predictedPrice)}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-background/50 border-border">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      {prediction.trend === 'up' ? (
                        <TrendingUp className="h-4 w-4 text-success" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-destructive" />
                      )}
                      <div>
                        <p className="text-sm text-muted-foreground">Trend</p>
                        <div className="flex items-center space-x-2">
                          <p className={`text-xl font-bold ${
                            prediction.trend === 'up' ? 'text-success' : 'text-destructive'
                          }`}>
                            {prediction.trend === 'up' ? 'Up' : 'Down'}
                          </p>
                          <span className="text-sm text-muted-foreground">
                            ({prediction.confidence}% confidence)
                          </span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {chartData.length > 0 && (
                <Card className="bg-background/50 border-border">
                  <CardHeader>
                    <CardTitle className="text-foreground text-sm">Prediction Overlay</CardTitle>
                    <CardDescription className="text-muted-foreground">
                      Last 30 days + predicted next day
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis 
                            dataKey="date" 
                            stroke="hsl(var(--muted-foreground))"
                            fontSize={12}
                            tickFormatter={(value) => new Date(value).toLocaleDateString()}
                          />
                          <YAxis 
                            stroke="hsl(var(--muted-foreground))"
                            fontSize={12}
                            tickFormatter={formatCurrency}
                          />
                          <Tooltip 
                            contentStyle={{
                              backgroundColor: 'hsl(var(--popover))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
                              color: 'hsl(var(--foreground))'
                            }}
                            labelFormatter={(value) => new Date(value).toLocaleDateString()}
                            formatter={(value, name, props) => [
                              formatCurrency(Number(value)), 
                              props.payload.isPredicted ? 'Predicted Price' : 'Close Price'
                            ]}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="close" 
                            stroke="hsl(var(--primary))" 
                            strokeWidth={2}
                            dot={(props) => {
                              if (props.payload.isPredicted) {
                                return (
                                  <circle 
                                    cx={props.cx} 
                                    cy={props.cy} 
                                    r={6} 
                                    fill="hsl(var(--warning))"
                                    stroke="hsl(var(--warning))"
                                    strokeWidth={2}
                                    className="animate-pulse-glow"
                                  />
                                );
                              }
                              return null;
                            }}
                          />
                          {/* Predicted portion as separate line */}
                          {prediction && (
                            <Line 
                              type="monotone" 
                              dataKey="close" 
                              stroke="hsl(var(--warning))" 
                              strokeWidth={2}
                              strokeDasharray="5 5"
                              connectNulls={false}
                              data={chartData.filter(d => d.isPredicted)}
                              dot={false}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictionPanel;