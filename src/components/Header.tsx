import { Button } from "@/components/ui/button";
import { TrendingUp } from "lucide-react";

const Header = () => {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="p-2 bg-gradient-primary rounded-lg shadow-primary">
              <TrendingUp className="h-6 w-6 text-primary-foreground" />
            </div>
            <h1 className="text-xl font-bold text-foreground">Stock Predictor MVP</h1>
          </div>
          
          <nav className="hidden md:flex items-center space-x-6">
            <Button variant="ghost" className="text-foreground hover:text-primary">
              Home
            </Button>
            <Button variant="ghost" className="text-foreground hover:text-primary">
              Predict
            </Button>
            <Button variant="ghost" className="text-foreground hover:text-primary">
              About
            </Button>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;