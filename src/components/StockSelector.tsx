import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon, TrendingUp } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

interface StockSelectorProps {
  onFetchData: (ticker: string, startDate: Date, endDate: Date) => void;
  isLoading?: boolean;
}

const StockSelector = ({ onFetchData, isLoading }: StockSelectorProps) => {
  const [selectedTicker, setSelectedTicker] = useState<string>("");
  const [startDate, setStartDate] = useState<Date>();
  const [endDate, setEndDate] = useState<Date>(new Date());

  const stocks = [
    { symbol: "AAPL", name: "Apple Inc." },
    { symbol: "MSFT", name: "Microsoft Corporation" },
    { symbol: "GOOGL", name: "Alphabet Inc." },
    { symbol: "AMZN", name: "Amazon.com Inc." },
    { symbol: "TSLA", name: "Tesla Inc." },
    { symbol: "NVDA", name: "NVIDIA Corporation" },
    { symbol: "META", name: "Meta Platforms Inc." },
    { symbol: "NFLX", name: "Netflix Inc." },
  ];

  const handleFetchData = () => {
    if (selectedTicker && startDate && endDate) {
      onFetchData(selectedTicker, startDate, endDate);
    }
  };

  return (
    <Card className="bg-gradient-card shadow-card border-border">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          <CardTitle className="text-foreground">Stock Selection</CardTitle>
        </div>
        <CardDescription className="text-muted-foreground">
          Choose a stock ticker and date range to analyze
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Stock Ticker</label>
          <Select value={selectedTicker} onValueChange={setSelectedTicker}>
            <SelectTrigger className="bg-background border-border">
              <SelectValue placeholder="Select a stock..." />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border">
              {stocks.map((stock) => (
                <SelectItem key={stock.symbol} value={stock.symbol}>
                  <div className="flex items-center justify-between w-full">
                    <span className="font-medium">{stock.symbol}</span>
                    <span className="text-muted-foreground text-sm ml-2">{stock.name}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Start Date</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full justify-start text-left font-normal bg-background border-border",
                    !startDate && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {startDate ? format(startDate, "PPP") : "Pick start date"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0 bg-popover border-border" align="start">
                <Calendar
                  mode="single"
                  selected={startDate}
                  onSelect={setStartDate}
                  initialFocus
                  className="p-3 pointer-events-auto"
                />
              </PopoverContent>
            </Popover>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">End Date</label>
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full justify-start text-left font-normal bg-background border-border",
                    !endDate && "text-muted-foreground"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {endDate ? format(endDate, "PPP") : "Pick end date"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0 bg-popover border-border" align="start">
                <Calendar
                  mode="single"
                  selected={endDate}
                  onSelect={setEndDate}
                  initialFocus
                  className="p-3 pointer-events-auto"
                />
              </PopoverContent>
            </Popover>
          </div>
        </div>

        <Button 
          onClick={handleFetchData} 
          disabled={!selectedTicker || !startDate || !endDate || isLoading}
          className="w-full bg-gradient-primary hover:opacity-90 text-primary-foreground shadow-primary"
        >
          {isLoading ? "Fetching Data..." : "Fetch Data"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default StockSelector;