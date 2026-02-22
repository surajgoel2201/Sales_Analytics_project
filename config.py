"""
Configuration file for Sales Analytics Project
"""

from datetime import datetime, timedelta

# =============================================================================
# COMPANY CONFIGURATION
# =============================================================================

# Major retail companies to analyze (using stock tickers as proxy for sales data)
COMPANIES = {
    'WMT': 'Walmart Inc.',
    'AMZN': 'Amazon.com Inc.',
    'COST': 'Costco Wholesale Corporation',
    'TGT': 'Target Corporation',
    'HD': 'Home Depot Inc.'
}

# Logging settings
LOG_LEVEL = "INFO"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Default stock settings
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

# Example companies
COMPANIES = {
    "WMT": "Walmart",
    "AMZN": "Amazon",
    "AAPL": "Apple"
}

# Data folders
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Default ticker for quick analysis
DEFAULT_TICKER = 'WMT'

# =============================================================================
# DATE CONFIGURATION
# =============================================================================

# Analysis period
START_DATE = '2023-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Historical data period options
PERIODS = {
    '1mo': '1 Month',
    '3mo': '3 Months',
    '6mo': '6 Months',
    '1y': '1 Year',
    '2y': '2 Years',
    '5y': '5 Years',
    'max': 'Maximum Available'
}

DEFAULT_PERIOD = '1y'

# =============================================================================
# TECHNICAL INDICATORS CONFIGURATION
# =============================================================================

# Simple Moving Averages
SMA_SHORT = 50   # 50-day moving average
SMA_LONG = 200   # 200-day moving average

# Exponential Moving Averages
EMA_SHORT = 12   # 12-day EMA
EMA_LONG = 26    # 26-day EMA

# Bollinger Bands
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# RSI (Relative Strength Index)
RSI_PERIOD = 14

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Column names mapping
COLUMN_MAPPING = {
    'Open': 'opening_price',
    'High': 'high_price',
    'Low': 'low_price',
    'Close': 'closing_price',
    'Volume': 'sales_volume',
    'Adj Close': 'adjusted_close'
}

# Columns to use for analysis
ANALYSIS_COLUMNS = ['closing_price', 'sales_volume', 'high_price', 'low_price']

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Chart styling
CHART_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (14, 8)
DPI = 100

# Color scheme
COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# Company colors for visualization
COMPANY_COLORS = {
    'WMT': '#0071ce',   # Walmart blue
    'AMZN': '#ff9900',  # Amazon orange
    'COST': '#0066b2',  # Costco blue
    'TGT': '#cc0000',   # Target red
    'HD': '#f96302'     # Home Depot orange
}

# =============================================================================
# FILE PATHS
# =============================================================================

# Directory structure
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'
REPORTS_DIR = 'reports'
FIGURES_DIR = f'{REPORTS_DIR}/figures'
NOTEBOOKS_DIR = 'notebooks'

# Report file names
HTML_REPORT = f'{REPORTS_DIR}/sales_report.html'
CSV_EXPORT = f'{REPORTS_DIR}/sales_analysis.csv'

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Metrics to calculate
METRICS = [
    'total_volume',
    'average_price',
    'price_change',
    'price_change_pct',
    'volatility',
    'volume_trend',
    'momentum'
]

# Alert thresholds
ALERT_THRESHOLDS = {
    'price_drop': -5.0,      # Alert if price drops more than 5%
    'price_spike': 10.0,     # Alert if price spikes more than 10%
    'volume_spike': 2.0,     # Alert if volume is 2x average
    'volatility_high': 0.05  # Alert if volatility exceeds 5%
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

# yfinance settings
YFINANCE_TIMEOUT = 30  # seconds
YFINANCE_RETRY_ATTEMPTS = 3
YFINANCE_RETRY_DELAY = 2  # seconds

# Data interval options
INTERVALS = ['1d', '1wk', '1mo']
DEFAULT_INTERVAL = '1d'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'sales_analytics.log'

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# Excel export settings
EXCEL_SHEET_NAMES = {
    'summary': 'Executive Summary',
    'raw_data': 'Raw Data',
    'metrics': 'Key Metrics',
    'comparison': 'Company Comparison'
}

# CSV delimiter
CSV_DELIMITER = ','

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database type: 'sqlite', 'postgresql', 'mysql'
DATABASE_TYPE = 'sqlite'

# SQLite database file path
SQLITE_DB_PATH = 'sales_analytics.db'

# PostgreSQL configuration (if using PostgreSQL)
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'sales_analytics',
    'user': 'postgres',
    'password': 'your_password'
}

# MySQL configuration (if using MySQL)
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'database': 'sales_analytics',
    'user': 'root',
    'password': 'your_password'
}

# Database options
DB_ENABLE_CACHING = True
DB_AUTO_COMMIT = True
DB_POOL_SIZE = 5

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Enable/disable features
ENABLE_CACHING = True
ENABLE_LOGGING = True
ENABLE_NOTIFICATIONS = False
ENABLE_AUTO_REFRESH = False
ENABLE_DATABASE = True  # Enable database storage

# Auto-refresh interval (minutes)
AUTO_REFRESH_INTERVAL = 15

# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
HOURS_PER_TRADING_DAY = 6.5
