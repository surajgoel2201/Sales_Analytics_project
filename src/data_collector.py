"""
Data Collection Module
Fetches live financial data from Yahoo Finance
"""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import config


# Setup logging


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL,logging.INFO),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Collects financial data from Yahoo Finance API
    """
    
    def __init__(self):
        """Initialize the data collector"""
        self.companies = config.COMPANIES
        logger.info("StockDataCollector initialized")
    
    def get_stock_data(
        self, 
        ticker: str, 
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """
        Fetch stock data for a single ticker
        
        Args:
            ticker: Stock ticker symbol (e.g., 'WMT', 'AMZN')
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Fetching data for {ticker} - Period: {period}, Interval: {interval}")
            
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Add company name if available
            if ticker in self.companies:
                df['Company'] = self.companies[ticker]
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self,
        tickers: List[str],
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple tickers
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period
            interval: Data interval
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        
        for ticker in tickers:
            df = self.get_stock_data(ticker, period, interval)
            if not df.empty:
                results[ticker] = df
        
        logger.info(f"Fetched data for {len(results)} out of {len(tickers)} tickers")
        return results
    
    def get_combined_data(
        self,
        tickers: Optional[List[str]] = None,
        period: str = config.DEFAULT_PERIOD,
        interval: str = config.DEFAULT_INTERVAL
    ) -> pd.DataFrame:
        """
        Fetch and combine data for multiple tickers into single DataFrame
        
        Args:
            tickers: List of stock ticker symbols (uses config.COMPANIES if None)
            period: Time period
            interval: Data interval
            
        Returns:
            Combined DataFrame with all ticker data
        """
        if tickers is None:
            tickers = list(self.companies.keys())
        
        logger.info(f"Fetching combined data for {len(tickers)} tickers")
        
        all_data = []
        
        for ticker in tickers:
            df = self.get_stock_data(ticker, period, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            logger.warning("No data retrieved for any ticker")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, axis=0)
        combined = combined.sort_index()
        
        logger.info(f"Combined data shape: {combined.shape}")
        return combined
    
    def get_company_info(self, ticker: str) -> Dict:
        """
        Get detailed company information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant business metrics
            company_data = {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'revenue': info.get('totalRevenue', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'description': info.get('longBusinessSummary', 'N/A')
            }
            
            return company_data
            
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {}
    
    def get_latest_price(self, ticker: str) -> float:
        """
        Get the most recent closing price
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Latest closing price
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            
            if not data.empty:
                return data['Close'].iloc[-1]
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching latest price for {ticker}: {str(e)}")
            return 0.0
    
    def save_data(self, df: pd.DataFrame, filename: str, data_type: str = 'raw'):
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            data_type: 'raw' or 'processed' (determines subdirectory)
        """
        try:
            if data_type == 'raw':
                filepath = f"{config.RAW_DATA_DIR}/{filename}"
            else:
                filepath = f"{config.PROCESSED_DATA_DIR}/{filename}"
            
            df.to_csv(filepath)
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def get_realtime_quote(self, ticker: str) -> Dict:
        """
        Get real-time quote data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            quote = {
                'ticker': ticker,
                'price': info.get('currentPrice', 0),
                'open': info.get('open', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0)
            }
            
            return quote
            
        except Exception as e:
            logger.error(f"Error fetching realtime quote for {ticker}: {str(e)}")
            return {}


if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    
    # Fetch single stock data
    walmart_data = collector.get_stock_data('WMT', period='1y')
    print(f"Walmart data shape: {walmart_data.shape}")
    print(walmart_data.head())
    
    # Fetch multiple stocks
    combined_data = collector.get_combined_data()
    print(f"\nCombined data shape: {combined_data.shape}")
    
    # Get company info
    info = collector.get_company_info('WMT')
    print(f"\nWalmart Info: {info}")

