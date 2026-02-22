"""
Visualization Module
Creates charts, graphs, and dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
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

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')


class DashboardGenerator:
    """
    Generates visualizations and dashboards
    """
    
    def __init__(self):
        """Initialize the dashboard generator"""
        logger.info("DashboardGenerator initialized")
    
    def plot_price_trend(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        save_path: Optional[str] = None
    ):
        """
        Plot price trend with moving averages
        
        Args:
            df: DataFrame with price data
            ticker: Stock ticker symbol
            save_path: Path to save figure (optional)
        """
        logger.info(f"Plotting price trend for {ticker}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE, 
                                        height_ratios=[3, 1])
        
        # Price plot
        ax1.plot(df.index, df['Close'], label='Close Price', 
                linewidth=2, color=config.COLOR_PALETTE['primary'])
        
        if f'SMA_{config.SMA_SHORT}' in df.columns:
            ax1.plot(df.index, df[f'SMA_{config.SMA_SHORT}'], 
                    label=f'{config.SMA_SHORT}-day SMA',
                    linestyle='--', alpha=0.7, color=config.COLOR_PALETTE['secondary'])
        
        if f'SMA_{config.SMA_LONG}' in df.columns:
            ax1.plot(df.index, df[f'SMA_{config.SMA_LONG}'], 
                    label=f'{config.SMA_LONG}-day SMA',
                    linestyle='--', alpha=0.7, color=config.COLOR_PALETTE['danger'])
        
        ax1.set_title(f'{ticker} - Price Trend Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        plt.close()
    
    def plot_candlestick(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        save_path: Optional[str] = None
    ):
        """
        Create candlestick chart using Plotly
        
        Args:
            df: DataFrame with OHLC data
            ticker: Stock ticker symbol
            save_path: Path to save HTML (optional)
        """
        logger.info(f"Creating candlestick chart for {ticker}")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if f'SMA_{config.SMA_SHORT}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{config.SMA_SHORT}'],
                    name=f'SMA {config.SMA_SHORT}',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{ticker} - Candlestick Chart',
            yaxis_title='Price ($)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive chart saved to {save_path}")
        
        return fig
    
    def plot_multiple_stocks(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ):
        """
        Plot multiple stocks for comparison
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames
            save_path: Path to save figure (optional)
        """
        logger.info(f"Plotting comparison for {len(data_dict)} stocks")
        
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
        
        for ticker, df in data_dict.items():
            # Normalize to starting price for comparison
            normalized = (df['Close'] / df['Close'].iloc[0]) * 100
            
            color = config.COMPANY_COLORS.get(ticker, None)
            ax.plot(df.index, normalized, label=ticker, linewidth=2, color=color)
        
        ax.set_title('Stock Performance Comparison (Normalized)', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Relative Performance (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Comparison chart saved to {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ):
        """
        Create correlation heatmap for multiple stocks
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames
            save_path: Path to save figure (optional)
        """
        logger.info("Creating correlation heatmap")
        
        # Create DataFrame with closing prices
        prices = pd.DataFrame()
        for ticker, df in data_dict.items():
            prices[ticker] = df['Close']
        
        # Calculate correlation matrix
        corr = prices.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax)
        
        ax.set_title('Stock Price Correlation Matrix', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        plt.close()
    
    def plot_volume_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        save_path: Optional[str] = None
    ):
        """
        Create volume analysis visualization
        
        Args:
            df: DataFrame with volume data
            ticker: Stock ticker symbol
            save_path: Path to save figure (optional)
        """
        logger.info(f"Creating volume analysis for {ticker}")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE)
        
        # Volume over time
        ax1.bar(df.index, df['Volume'], alpha=0.7, 
               color=config.COLOR_PALETTE['primary'])
        if 'Volume_MA' in df.columns:
            ax1.plot(df.index, df['Volume_MA'], 
                    color=config.COLOR_PALETTE['danger'],
                    linewidth=2, label='Volume MA')
            ax1.legend()
        
        ax1.set_title(f'{ticker} - Volume Analysis', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Volume', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Volume distribution
        ax2.hist(df['Volume'], bins=50, alpha=0.7, 
                color=config.COLOR_PALETTE['secondary'])
        ax2.set_xlabel('Volume', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Volume Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Volume analysis saved to {save_path}")
        
        plt.close()
    
    def create_performance_dashboard(
        self,
        data_dict: Dict[str, pd.DataFrame],
        metrics_dict: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive performance dashboard
        
        Args:
            data_dict: Dictionary mapping tickers to DataFrames
            metrics_dict: Dictionary mapping tickers to metrics
            save_path: Path to save HTML (optional)
        """
        logger.info("Creating performance dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price Comparison',
                'Volume Comparison',
                'Daily Returns Distribution',
                'Performance Metrics',
                'Volatility Analysis',
                'Key Statistics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "table"}],
                [{"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. Price comparison (normalized)
        for ticker, df in data_dict.items():
            normalized = (df['Close'] / df['Close'].iloc[0]) * 100
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized,
                    name=ticker,
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # 2. Volume comparison
        volumes = [metrics_dict[ticker].get('total_volume', 0) 
                  for ticker in data_dict.keys()]
        fig.add_trace(
            go.Bar(
                x=list(data_dict.keys()),
                y=volumes,
                name='Total Volume',
                marker_color=[config.COMPANY_COLORS.get(t, 'blue') 
                             for t in data_dict.keys()]
            ),
            row=1, col=2
        )
        
        # 3. Returns distribution (first stock)
        first_ticker = list(data_dict.keys())[0]
        first_df = data_dict[first_ticker]
        if 'Daily_Return' in first_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=first_df['Daily_Return'] * 100,
                    name='Daily Returns',
                    nbinsx=50
                ),
                row=2, col=1
            )
        
        # 4. Performance metrics table
        metrics_data = []
        for ticker, metrics in metrics_dict.items():
            metrics_data.append([
                ticker,
                f"${metrics.get('current_price', 0):.2f}",
                f"{metrics.get('price_change_pct', 0):.2f}%",
                f"{metrics.get('daily_volatility', 0):.2f}%"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Ticker', 'Price', 'Change %', 'Volatility']),
                cells=dict(values=list(zip(*metrics_data)))
            ),
            row=2, col=2
        )
        
        # 5. Volatility comparison
        volatilities = [metrics_dict[ticker].get('daily_volatility', 0) 
                       for ticker in data_dict.keys()]
        fig.add_trace(
            go.Scatter(
                x=list(data_dict.keys()),
                y=volatilities,
                mode='markers+lines',
                name='Volatility',
                marker=dict(size=12)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Comprehensive Performance Dashboard"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def create_technical_analysis_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        save_path: Optional[str] = None
    ):
        """
        Create technical analysis chart with indicators
        
        Args:
            df: DataFrame with technical indicators
            ticker: Stock ticker symbol
            save_path: Path to save HTML (optional)
        """
        logger.info(f"Creating technical analysis chart for {ticker}")
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=(f'{ticker} Price with Bollinger Bands', 
                          'Volume', 'RSI', 'MACD')
        )
        
        # Price with Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], 
                          name='BB Upper', line=dict(dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Middle'], 
                          name='BB Middle', line=dict(dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], 
                          name='BB Lower', line=dict(dash='dash')),
                row=1, col=1
            )
        
        # Volume
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], 
                  marker_color=colors, name='Volume'),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], 
                          name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         row=3, col=1)
        
        fig.update_layout(
            height=1000,
            title_text=f"{ticker} - Technical Analysis",
            xaxis_rangeslider_visible=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Technical chart saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    from data_collector import StockDataCollector
    from data_processor import DataProcessor
    from analyzer import SalesAnalyzer
    
    collector = StockDataCollector()
    processor = DataProcessor()
    viz = DashboardGenerator()
    
    # Get data for multiple stocks
    tickers = ['WMT', 'AMZN', 'COST']
    data_dict = {}
    metrics_dict = {}
    
    for ticker in tickers:
        data = collector.get_stock_data(ticker, period='6mo')
        processed = processor.clean_data(data)
        processed = processor.add_technical_indicators(processed)
        
        analyzer = SalesAnalyzer(processed)
        metrics = analyzer.calculate_key_metrics()
        
        data_dict[ticker] = processed
        metrics_dict[ticker] = metrics
    
    # Create visualizations
    viz.plot_multiple_stocks(data_dict, 
                             save_path=f'{config.FIGURES_DIR}/comparison.png')
    viz.create_performance_dashboard(data_dict, metrics_dict,
                                    save_path=f'{config.REPORTS_DIR}/dashboard.html')
    
    print("Visualizations created successfully!")
