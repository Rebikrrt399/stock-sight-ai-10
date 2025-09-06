import { useState } from "react";
import Header from "@/components/Header";
import StockSelector from "@/components/StockSelector";
import StockChart from "@/components/StockChart";
import PredictionPanel from "@/components/PredictionPanel";
import { useToast } from "@/hooks/use-toast";
import heroImage from "@/assets/hero-financial.jpg";

interface StockData {
  date: string;
  close: number;
  volume: number;
  high: number;
  low: number;
}

const Index = () => {
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // Mock data generator for demonstration
  const generateMockData = (ticker: string, startDate: Date, endDate: Date): StockData[] => {
    const data: StockData[] = [];
    const basePrice = Math.random() * 200 + 50; // Random base price between 50-250
    const days = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
    
    for (let i = 0; i <= days; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      const randomChange = (Math.random() - 0.5) * 0.1; // ±5% change
      const price = basePrice * (1 + randomChange * i * 0.01);
      
      data.push({
        date: date.toISOString().split('T')[0],
        close: Math.max(price + (Math.random() - 0.5) * 10, 1),
        volume: Math.floor(Math.random() * 50000000) + 1000000,
        high: price + Math.random() * 5,
        low: price - Math.random() * 5,
      });
    }
    
    return data;
  };

  const handleFetchData = async (ticker: string, startDate: Date, endDate: Date) => {
    setIsLoading(true);
    setSelectedTicker(ticker);
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const mockData = generateMockData(ticker, startDate, endDate);
      setStockData(mockData);
      
      toast({
        title: "Data Loaded Successfully",
        description: `Fetched ${mockData.length} days of data for ${ticker}`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch stock data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    // Simulate AI prediction
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const lastPrice = stockData[stockData.length - 1]?.close || 100;
    const change = (Math.random() - 0.5) * 0.1; // ±5% change
    const predictedPrice = lastPrice * (1 + change);
    
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    return {
      predictedPrice,
      trend: change > 0 ? 'up' as const : 'down' as const,
      confidence: Math.floor(Math.random() * 30) + 70, // 70-100% confidence
      predictedDate: tomorrow.toISOString().split('T')[0],
    };
  };

  const calculateStats = (data: StockData[]) => {
    if (data.length === 0) return undefined;
    
    const prices = data.map(d => d.close);
    return {
      avgClose: prices.reduce((a, b) => a + b, 0) / prices.length,
      highestPrice: Math.max(...prices),
      lowestPrice: Math.min(...prices),
    };
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Hero Section */}
      <section className="relative h-96 overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${heroImage})` }}
        />
        <div className="absolute inset-0 bg-background/80" />
        <div className="relative container mx-auto px-4 h-full flex items-center justify-center">
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-6xl font-bold text-foreground">
              AI-Powered Stock Prediction
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl">
              Analyze historical data and get intelligent predictions for your favorite stocks
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 space-y-8">
        <StockSelector onFetchData={handleFetchData} isLoading={isLoading} />
        
        {stockData.length > 0 && (
          <>
            <StockChart 
              data={stockData} 
              ticker={selectedTicker}
              stats={calculateStats(stockData)}
            />
            
            <PredictionPanel 
              ticker={selectedTicker}
              currentData={stockData}
              onPredict={handlePredict}
              isLoading={isLoading}
            />
          </>
        )}
      </main>
    </div>
  );
};

export default Index;