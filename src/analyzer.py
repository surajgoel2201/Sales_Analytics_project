"""
Analysis Module
Performs business analytics on financial data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class SalesAnalyzer:
    """
    Analyzes sales performance metrics from financial data
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with data
        
        Args:
            data: DataFrame with financial data
        """
        self.data = data
        logger.info(f"SalesAnalyzer initialized with {len(data)} records")
    
    def calculate_key_metrics(self) -> Dict:
        """
        Calculate key performance metrics
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating key metrics")
        
        metrics = {}
        
        # Price metrics
        if 'Close' in self.data.columns:
            metrics['current_price'] = self.data['Close'].iloc[-1]
            metrics['avg_price'] = self.data['Close'].mean()
            metrics['max_price'] = self.data['Close'].max()
            metrics['min_price'] = self.data['Close'].min()
            metrics['price_range'] = metrics['max_price'] - metrics['min_price']
            
            # Price changes
            if len(self.data) > 1:
                metrics['price_change'] = (
                    self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]
                )
                metrics['price_change_pct'] = (
                    metrics['price_change'] / self.data['Close'].iloc[0] * 100
                )
        
        # Volume metrics (proxy for sales activity)
        if 'Volume' in self.data.columns:
            metrics['total_volume'] = self.data['Volume'].sum()
            metrics['avg_volume'] = self.data['Volume'].mean()
            metrics['max_volume'] = self.data['Volume'].max()
            metrics['volume_volatility'] = self.data['Volume'].std()
        
        # Volatility metrics
        if 'Daily_Return' in self.data.columns:
            metrics['avg_daily_return'] = self.data['Daily_Return'].mean() * 100
            metrics['daily_volatility'] = self.data['Daily_Return'].std() * 100
            metrics['annualized_volatility'] = (
                metrics['daily_volatility'] * np.sqrt(config.TRADING_DAYS_PER_YEAR)
            )
        
        # Trading range
        if all(col in self.data.columns for col in ['High', 'Low']):
            metrics['avg_daily_range'] = (
                (self.data['High'] - self.data['Low']).mean()
            )
        
        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics
    
    def analyze_trends(self) -> Dict:
        """
        Analyze price and volume trends
        
        Returns:
            Dictionary with trend analysis
        """
        logger.info("Analyzing trends")
        
        trends = {}
        
        if 'Close' in self.data.columns:
            # Overall trend direction
            first_price = self.data['Close'].iloc[0]
            last_price = self.data['Close'].iloc[-1]
            
            if last_price > first_price:
                trends['overall_trend'] = 'Upward'
                trends['trend_strength'] = (
                    (last_price - first_price) / first_price * 100
                )
            elif last_price < first_price:
                trends['overall_trend'] = 'Downward'
                trends['trend_strength'] = (
                    (first_price - last_price) / first_price * 100
                )
            else:
                trends['overall_trend'] = 'Flat'
                trends['trend_strength'] = 0
        
        # Moving average trends
        if f'SMA_{config.SMA_SHORT}' in self.data.columns:
            latest_sma_short = self.data[f'SMA_{config.SMA_SHORT}'].iloc[-1]
            latest_sma_long = self.data[f'SMA_{config.SMA_LONG}'].iloc[-1]
            
            if latest_sma_short > latest_sma_long:
                trends['ma_trend'] = 'Bullish (Golden Cross)'
            else:
                trends['ma_trend'] = 'Bearish (Death Cross)'
        
        # Volume trend
        if 'Volume' in self.data.columns:
            recent_volume = self.data['Volume'].tail(30).mean()
            earlier_volume = self.data['Volume'].head(30).mean()
            
            if recent_volume > earlier_volume:
                trends['volume_trend'] = 'Increasing'
            else:
                trends['volume_trend'] = 'Decreasing'
        
        return trends
    
    def identify_patterns(self) -> List[str]:
        """
        Identify common chart patterns
        
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if len(self.data) < 50:
            return patterns
        
        # Check for breakouts
        recent_high = self.data['Close'].tail(20).max()
        historical_high = self.data['Close'].iloc[:-20].max()
        
        if recent_high > historical_high:
            patterns.append("Price Breakout")
        
        # Check for support/resistance
        current_price = self.data['Close'].iloc[-1]
        price_std = self.data['Close'].std()
        
        # Volume spike
        if 'Volume' in self.data.columns:
            recent_volume = self.data['Volume'].iloc[-1]
            avg_volume = self.data['Volume'].mean()
            
            if recent_volume > avg_volume * 2:
                patterns.append("Volume Spike")
        
        # RSI patterns
        if 'RSI' in self.data.columns:
            current_rsi = self.data['RSI'].iloc[-1]
            
            if current_rsi > 70:
                patterns.append("Overbought (RSI > 70)")
            elif current_rsi < 30:
                patterns.append("Oversold (RSI < 30)")
        
        return patterns
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate advanced performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Calculating performance metrics")
        
        metrics = {}
        
        if 'Daily_Return' in self.data.columns:
            returns = self.data['Daily_Return'].dropna()
            
            # Sharpe Ratio (simplified, assuming 0% risk-free rate)
            if returns.std() != 0:
                metrics['sharpe_ratio'] = (
                    returns.mean() / returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
                )
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min() * 100
            
            # Win Rate
            winning_days = (returns > 0).sum()
            total_days = len(returns)
            metrics['win_rate'] = (winning_days / total_days * 100) if total_days > 0 else 0
        
        # Price momentum
        if 'Close' in self.data.columns and len(self.data) >= 20:
            metrics['momentum_20d'] = (
                (self.data['Close'].iloc[-1] / self.data['Close'].iloc[-20] - 1) * 100
            )
        
        return metrics
    
    def compare_to_benchmark(self, benchmark_data: pd.DataFrame) -> Dict:
        """
        Compare performance to a benchmark
        
        Args:
            benchmark_data: DataFrame with benchmark data
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        if 'Close' not in self.data.columns or 'Close' not in benchmark_data.columns:
            return comparison
        
        # Align data
        common_dates = self.data.index.intersection(benchmark_data.index)
        
        if len(common_dates) == 0:
            return comparison
        
        stock_returns = self.data.loc[common_dates, 'Close'].pct_change()
        benchmark_returns = benchmark_data.loc[common_dates, 'Close'].pct_change()
        
        # Beta calculation
        covariance = stock_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        
        if benchmark_variance != 0:
            comparison['beta'] = covariance / benchmark_variance
        
        # Alpha calculation (simplified)
        stock_total_return = (
            (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
        )
        benchmark_total_return = (
            (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0] - 1) * 100
        )
        
        comparison['alpha'] = stock_total_return - benchmark_total_return
        comparison['stock_return'] = stock_total_return
        comparison['benchmark_return'] = benchmark_total_return
        
        return comparison
    
    def generate_insights(self) -> List[str]:
        """
        Generate human-readable insights from the data
        
        Returns:
            List of insight strings
        """
        insights = []
        
        metrics = self.calculate_key_metrics()
        trends = self.analyze_trends()
        patterns = self.identify_patterns()
        
        # Price insights
        if 'price_change_pct' in metrics:
            change = metrics['price_change_pct']
            if abs(change) > 10:
                direction = "increased" if change > 0 else "decreased"
                insights.append(
                    f"Significant price movement: {direction} by {abs(change):.2f}% over the period"
                )
        
        # Trend insights
        if 'overall_trend' in trends:
            insights.append(f"Overall trend: {trends['overall_trend']}")
        
        if 'ma_trend' in trends:
            insights.append(f"Moving average signal: {trends['ma_trend']}")
        
        # Pattern insights
        for pattern in patterns:
            insights.append(f"Pattern detected: {pattern}")
        
        # Volume insights
        if 'total_volume' in metrics and 'avg_volume' in metrics:
            recent_vol = self.data['Volume'].tail(10).mean()
            avg_vol = metrics['avg_volume']
            
            if recent_vol > avg_vol * 1.5:
                insights.append("Recent trading activity is significantly higher than average")
            elif recent_vol < avg_vol * 0.5:
                insights.append("Recent trading activity is lower than average")
        
        # Volatility insights
        if 'daily_volatility' in metrics:
            vol = metrics['daily_volatility']
            if vol > 2:
                insights.append("High volatility detected - prices are fluctuating significantly")
            elif vol < 0.5:
                insights.append("Low volatility - prices are relatively stable")
        
        return insights
    
    def create_summary_report(self) -> Dict:
        """
        Create a comprehensive summary report
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Creating summary report")
        
        report = {
            'metrics': self.calculate_key_metrics(),
            'trends': self.analyze_trends(),
            'patterns': self.identify_patterns(),
            'performance': self.calculate_performance_metrics(),
            'insights': self.generate_insights()
        }
        
        return report


if __name__ == "__main__":
    # Example usage
    from data_collector import StockDataCollector
    from data_processor import DataProcessor
    
    collector = StockDataCollector()
    processor = DataProcessor()
    
    # Get and process data
    data = collector.get_stock_data('WMT', period='1y')
    processed = processor.clean_data(data)
    processed = processor.add_technical_indicators(processed)
    
    # Analyze
    analyzer = SalesAnalyzer(processed)
    report = analyzer.create_summary_report()
    
    print("=== ANALYSIS REPORT ===")
    print("\nKey Metrics:")
    for key, value in report['metrics'].items():
        print(f"  {key}: {value}")
    
    print("\nInsights:")
    for insight in report['insights']:
        print(f"  - {insight}")
