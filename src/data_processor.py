"""
Data Processing Module
Cleans and transforms raw financial data
"""

import sys
from pathlib import Path

# Fix import path BEFORE importing config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
import config

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and transforms financial data"""

    def __init__(self):
        logger.info("DataProcessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Cleaning data - Initial shape: {df.shape}")

        cleaned = df.copy()

        # Remove duplicate index rows
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

        # Price columns that exist
        price_columns = [
            col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']
            if col in cleaned.columns
        ]

        # Forward fill prices
        for col in price_columns:
            cleaned[col] = cleaned[col].ffill()

        # Volume fill
        if 'Volume' in cleaned.columns:
            cleaned['Volume'] = cleaned['Volume'].fillna(0)

        # Drop rows where all prices missing
        if price_columns:
            cleaned = cleaned.dropna(subset=price_columns, how='all')

        logger.info(f"Cleaning complete - Final shape: {cleaned.shape}")
        return cleaned

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding technical indicators")

        df = df.copy()

        if 'Close' not in df.columns:
            logger.warning("Close column missing, skipping indicators")
            return df

        # SMA
        df[f'SMA_{config.SMA_SHORT}'] = df['Close'].rolling(
            config.SMA_SHORT).mean()
        df[f'SMA_{config.SMA_LONG}'] = df['Close'].rolling(
            config.SMA_LONG).mean()

        # EMA
        df[f'EMA_{config.EMA_SHORT}'] = df['Close'].ewm(
            span=config.EMA_SHORT, adjust=False).mean()
        df[f'EMA_{config.EMA_LONG}'] = df['Close'].ewm(
            span=config.EMA_LONG, adjust=False).mean()

        # Bollinger
        sma = df['Close'].rolling(config.BOLLINGER_WINDOW).mean()
        std = df['Close'].rolling(config.BOLLINGER_WINDOW).std()

        df['BB_Upper'] = sma + std * config.BOLLINGER_STD
        df['BB_Lower'] = sma - std * config.BOLLINGER_STD
        df['BB_Middle'] = sma

        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], config.RSI_PERIOD)

        # Returns
        df['Daily_Return'] = df['Close'].pct_change()

        df['Volatility'] = df['Daily_Return'].rolling(
            config.BOLLINGER_WINDOW).std()

        # Volume MA safe
        if 'Volume' in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(
                config.SMA_SHORT).mean()

        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100

        logger.info("Indicators added")
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int):
        delta = prices.diff()

        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    

    def calculate_growth_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth metrics over different periods.

        Args:
            df: DataFrame with closing price data

        Returns:
            DataFrame with growth metrics added
        """
        logger.info("Calculating growth metrics")

        df = df.copy()

        if 'Close' not in df.columns:
            logger.warning("Close column missing, skipping growth metrics")
            return df

        # Month-over-Month (~21 trading days)
        df['MoM_Growth'] = df['Close'].pct_change(periods=21) * 100

        # Quarter-over-Quarter (~63 trading days)
        df['QoQ_Growth'] = df['Close'].pct_change(periods=63) * 100

        # Year-over-Year (~252 trading days)
        df['YoY_Growth'] = df['Close'].pct_change(periods=252) * 100

        logger.info("Growth metrics added")
        return df
    
    def export_processed_data(self, df: pd.DataFrame, filename: str):
        try:
            path = Path(config.PROCESSED_DATA_DIR)
            path.mkdir(parents=True, exist_ok=True)

            filepath = path / filename
            df.to_csv(filepath)

            logger.info(f"Exported to {filepath}")

        except Exception as e:
            logger.error(f"Export error: {e}")


if __name__ == "__main__":
    from data_collector import StockDataCollector

    collector = StockDataCollector()
    processor = DataProcessor()

    data = collector.get_stock_data('WMT', period='1y')

    cleaned = processor.clean_data(data)
    with_indicators = processor.add_technical_indicators(cleaned)
    with_growth = processor.calculate_growth_metrics(with_indicators)
    print("Columns:", with_indicators.columns.tolist())
    print(with_indicators.tail())
