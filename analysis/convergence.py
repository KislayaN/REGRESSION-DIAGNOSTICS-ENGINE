class Converge_insights:
    def __init__(self):
        self.insights = []
        
    def analyze_convergence(self, cost_history_scaled, cost_history_unscaled=None):
        
        if cost_history_scaled[-1] == 0:
            self.insights.append("No convergence data available")
            
        if cost_history_scaled[-1] < cost_history_scaled[0]:
            self.insights.append("Gradient Descent is converging")
            
        if cost_history_unscaled:
            if cost_history_unscaled[-1] > cost_history_scaled[-1]:
                self.insights.append("Scaling improved convergence performance")
                
        if any(cost > 1e6 for cost in cost_history_scaled):
            self.insights.append("Possible Divergence Detected (Learning rate too high)")
            
        return self.insights