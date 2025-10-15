import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { TrendingUp, TrendingDown, Zap, Target, Loader2, Brain } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { PredictionPanelProps, PredictionResult } from "./types"; // Import the interface

const PredictionPanel = ({ ticker, currentData, onPredict, isLoading }: PredictionPanelProps) => {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predicting, setPredicting] = useState(false);

  const handlePredict = async () => {
    setPredicting(true);
    try {
      const result = await onPredict();
      setPrediction(result);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
    setPredicting(false);
  };

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;

  const chartData = currentData.length > 0
    ? [
        ...currentData.slice(-30),
        ...(prediction
          ? [
              {
                date: prediction.predictedDate,
                close: prediction.predictedPrice,
                isPredicted: true
              }
            ]
          : [])
      ]
    : [];

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
            size="lg"
          >
            {predicting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing Market Data...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Generate AI Prediction
              </>
            )}
          </Button>

          {prediction && (
            <div className="space-y-4 animate-slide-up">
              {/* Predicted Price & Trend */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Price Card */}
                <Card className="bg-gradient-to-br from-card to-card/80 border-accent/30">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Target className="h-4 w-4 text-primary" />
                        <div>
                          <p className="text-sm text-muted-foreground">Predicted Price</p>
                          <p className="text-xl font-bold text-foreground">
                            {formatCurrency(prediction.predictedPrice)}
                          </p>
                          <p className="text-xs text-muted-foreground">for {prediction.predictedDate}</p>
                        </div>
                      </div>
                      {prediction.trend === "up" ? (
                        <Badge variant="secondary" className="bg-success/20 text-success border-success/30">
                          <TrendingUp className="h-3 w-3 mr-1" />
                          Bullish
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-destructive/20 text-destructive border-destructive/30">
                          <TrendingDown className="h-3 w-3 mr-1" />
                          Bearish
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Confidence Card */}
                <Card className="bg-gradient-to-br from-card to-card/80 border-accent/30">
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Brain className="h-4 w-4 text-primary" />
                      <div className="flex-1">
                        <p className="text-sm text-muted-foreground">AI Confidence</p>
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-xl font-bold text-foreground">{prediction.confidence}%</span>
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            prediction.confidence >= 80 ? "bg-success/20 text-success" :
                            prediction.confidence >= 60 ? "bg-warning/20 text-warning" :
                            "bg-destructive/20 text-destructive"
                          }`}>
                            {prediction.confidence >= 80 ? "High" : prediction.confidence >= 60 ? "Medium" : "Low"}
                          </span>
                        </div>
                        <Progress value={prediction.confidence} className="h-2" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Chart */}
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
                              backgroundColor: "hsl(var(--popover))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: "8px",
                              color: "hsl(var(--foreground))",
                            }}
                            labelFormatter={(value) => new Date(value).toLocaleDateString()}
                            formatter={(value, name, props) => [
                              formatCurrency(Number(value)),
                              props.payload.isPredicted ? "Predicted Price" : "Close Price",
                            ]}
                          />
                          <Line
                            type="monotone"
                            dataKey="close"
                            stroke="hsl(var(--primary))"
                            strokeWidth={2}
                            dot={(props) =>
                              props.payload.isPredicted ? (
                                <circle
                                  cx={props.cx}
                                  cy={props.cy}
                                  r={6}
                                  fill="hsl(var(--warning))"
                                  stroke="hsl(var(--warning))"
                                  strokeWidth={2}
                                  className="animate-pulse-glow"
                                />
                              ) : null
                            }
                          />
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
