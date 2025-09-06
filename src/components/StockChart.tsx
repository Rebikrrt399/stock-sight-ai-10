import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from "lucide-react";

interface StockData {
  date: string;
  close: number;
  volume: number;
  high: number;
  low: number;
}

interface StockChartProps {
  data: StockData[];
  ticker: string;
  stats?: {
    avgClose: number;
    highestPrice: number;
    lowestPrice: number;
  };
}

const StockChart = ({ data, ticker, stats }: StockChartProps) => {
  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
  const formatVolume = (value: number) => {
    if (value >= 1000000000) return `${(value / 1000000000).toFixed(1)}B`;
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
    return value.toString();
  };

  const isPositiveTrend = data.length > 1 && data[data.length - 1].close > data[0].close;

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-gradient-card shadow-card border-border">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4 text-primary" />
                <div>
                  <p className="text-sm text-muted-foreground">Average Close</p>
                  <p className="text-lg font-semibold text-foreground">{formatCurrency(stats.avgClose)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-card shadow-card border-border">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-success" />
                <div>
                  <p className="text-sm text-muted-foreground">Highest Price</p>
                  <p className="text-lg font-semibold text-success">{formatCurrency(stats.highestPrice)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-card shadow-card border-border">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingDown className="h-4 w-4 text-destructive" />
                <div>
                  <p className="text-sm text-muted-foreground">Lowest Price</p>
                  <p className="text-lg font-semibold text-destructive">{formatCurrency(stats.lowestPrice)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Price Chart */}
      <Card className="bg-gradient-card shadow-card border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {isPositiveTrend ? (
                <TrendingUp className="h-5 w-5 text-success" />
              ) : (
                <TrendingDown className="h-5 w-5 text-destructive" />
              )}
              <CardTitle className="text-foreground">{ticker} Price History</CardTitle>
            </div>
            <div className={`flex items-center space-x-1 px-2 py-1 rounded-md ${
              isPositiveTrend ? 'bg-success/20 text-success' : 'bg-destructive/20 text-destructive'
            }`}>
              {isPositiveTrend ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              <span className="text-xs font-medium">
                {isPositiveTrend ? 'Uptrend' : 'Downtrend'}
              </span>
            </div>
          </div>
          <CardDescription className="text-muted-foreground">
            Historical closing prices over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
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
                  formatter={(value) => [formatCurrency(Number(value)), 'Close Price']}
                />
                <Line 
                  type="monotone" 
                  dataKey="close" 
                  stroke="hsl(var(--primary))" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: 'hsl(var(--primary))', fill: 'hsl(var(--primary))' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Volume Chart */}
      <Card className="bg-gradient-card shadow-card border-border">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            <CardTitle className="text-foreground">Trading Volume</CardTitle>
          </div>
          <CardDescription className="text-muted-foreground">
            Daily trading volume over time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
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
                  tickFormatter={formatVolume}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                    color: 'hsl(var(--foreground))'
                  }}
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value) => [formatVolume(Number(value)), 'Volume']}
                />
                <Bar 
                  dataKey="volume" 
                  fill="hsl(var(--muted-foreground))"
                  opacity={0.7}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default StockChart;