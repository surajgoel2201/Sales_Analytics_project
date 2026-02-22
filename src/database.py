"""
Database Module
Handles SQL database operations for storing and retrieving analytics data
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import config
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQL database operations for sales analytics
    """
    
    def __init__(self, db_path: str = 'sales_analytics.db'):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        logger.info(f"DatabaseManager initialized with database: {db_path}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def create_tables(self):
        """Create database tables from schema"""
        try:
            # Read schema file
            schema_path = 'database_schema.sql'
            if not os.path.exists(schema_path):
                logger.warning(f"Schema file not found: {schema_path}")
                return
            
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema
            cursor = self.conn.cursor()
            cursor.executescript(schema_sql)
            self.conn.commit()
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    # =========================================================================
    # Company Operations
    # =========================================================================
    
    def insert_company(self, ticker: str, name: str, **kwargs):
        """
        Insert or update company information
        
        Args:
            ticker: Stock ticker symbol
            name: Company name
            **kwargs: Additional fields (sector, industry, market_cap, etc.)
        """
        try:
            cursor = self.conn.cursor()
            
            fields = ['ticker', 'name']
            values = [ticker, name]
            
            # Add optional fields
            for key, value in kwargs.items():
                if key in ['sector', 'industry', 'market_cap', 'employees', 'description']:
                    fields.append(key)
                    values.append(value)
            
            placeholders = ', '.join(['?' for _ in values])
            fields_str = ', '.join(fields)
            
            sql = f"""
                INSERT OR REPLACE INTO companies ({fields_str})
                VALUES ({placeholders})
            """
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Company {ticker} inserted/updated")
            
        except Exception as e:
            logger.error(f"Error inserting company: {str(e)}")
            raise
    
    def get_company(self, ticker: str) -> Optional[Dict]:
        """
        Get company information
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information or None
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM companies WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting company: {str(e)}")
            return None
    
    # =========================================================================
    # Stock Data Operations
    # =========================================================================
    
    def insert_stock_data(self, df: pd.DataFrame, ticker: str):
        """
        Insert stock data from DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
        """
        try:
            # Prepare data
            data = df.copy()
            data['ticker'] = ticker
            data['date'] = data.index
            
            # Rename columns to match database schema
            column_mapping = {
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Adj Close': 'adjusted_close',
                'Volume': 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Select only required columns
            columns = ['ticker', 'date', 'open_price', 'high_price', 
                      'low_price', 'close_price', 'adjusted_close', 'volume']
            data = data[columns]
            
            # Insert into database
            data.to_sql('stock_data', self.conn, if_exists='append', 
                       index=False, method='multi')
            
            logger.info(f"Inserted {len(data)} stock data records for {ticker}")
            
        except Exception as e:
            logger.error(f"Error inserting stock data: {str(e)}")
            raise
    
    def get_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve stock data from database
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        try:
            sql = "SELECT * FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            
            sql += " ORDER BY date"
            
            df = pd.read_sql_query(sql, self.conn, params=params, 
                                  parse_dates=['date'])
            df = df.set_index('date')
            
            logger.info(f"Retrieved {len(df)} stock data records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving stock data: {str(e)}")
            return pd.DataFrame()
    
    # =========================================================================
    # Technical Indicators Operations
    # =========================================================================
    
    def insert_technical_indicators(self, df: pd.DataFrame, ticker: str):
        """
        Insert technical indicators from DataFrame
        
        Args:
            df: DataFrame with technical indicators
            ticker: Stock ticker symbol
        """
        try:
            data = df.copy()
            data['ticker'] = ticker
            data['date'] = data.index
            
            # Select indicator columns
            indicator_columns = ['ticker', 'date']
            
            # Map column names to database schema
            column_mapping = {
                'SMA_50': 'sma_50',
                'SMA_200': 'sma_200',
                'EMA_12': 'ema_12',
                'EMA_26': 'ema_26',
                'RSI': 'rsi',
                'BB_Upper': 'bollinger_upper',
                'BB_Middle': 'bollinger_middle',
                'BB_Lower': 'bollinger_lower',
                'Volatility': 'volatility'
            }
            
            for original, new in column_mapping.items():
                if original in data.columns:
                    data[new] = data[original]
                    indicator_columns.append(new)
            
            # Insert into database
            data[indicator_columns].to_sql(
                'technical_indicators', 
                self.conn, 
                if_exists='append',
                index=False, 
                method='multi'
            )
            
            logger.info(f"Inserted technical indicators for {ticker}")
            
        except Exception as e:
            logger.error(f"Error inserting technical indicators: {str(e)}")
            raise
    
    # =========================================================================
    # Performance Metrics Operations
    # =========================================================================
    
    def insert_performance_metrics(self, metrics: Dict, ticker: str, date: str):
        """
        Insert performance metrics
        
        Args:
            metrics: Dictionary of metrics
            ticker: Stock ticker symbol
            date: Date of metrics
        """
        try:
            cursor = self.conn.cursor()
            
            sql = """
                INSERT OR REPLACE INTO performance_metrics
                (ticker, date, daily_return, volatility_30d, price_change_pct, volume_ma)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            values = (
                ticker,
                date,
                metrics.get('avg_daily_return', None),
                metrics.get('daily_volatility', None),
                metrics.get('price_change_pct', None),
                metrics.get('avg_volume', None)
            )
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Inserted performance metrics for {ticker}")
            
        except Exception as e:
            logger.error(f"Error inserting performance metrics: {str(e)}")
            raise
    
    # =========================================================================
    # Analysis Results Operations
    # =========================================================================
    
    def insert_analysis_result(
        self, 
        ticker: str, 
        analysis_type: str,
        result_value: str,
        notes: Optional[str] = None
    ):
        """
        Insert analysis result
        
        Args:
            ticker: Stock ticker symbol
            analysis_type: Type of analysis
            result_value: Result value
            notes: Additional notes
        """
        try:
            cursor = self.conn.cursor()
            
            sql = """
                INSERT INTO analysis_results
                (ticker, analysis_date, analysis_type, result_value, notes)
                VALUES (?, ?, ?, ?, ?)
            """
            
            values = (
                ticker,
                datetime.now().strftime('%Y-%m-%d'),
                analysis_type,
                result_value,
                notes
            )
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Inserted analysis result for {ticker}")
            
        except Exception as e:
            logger.error(f"Error inserting analysis result: {str(e)}")
            raise
    
    # =========================================================================
    # Trend Operations
    # =========================================================================
    
    def insert_trend(
        self,
        ticker: str,
        trend_type: str,
        start_date: str,
        end_date: Optional[str] = None,
        strength: Optional[float] = None
    ):
        """
        Insert detected trend
        
        Args:
            ticker: Stock ticker symbol
            trend_type: Type of trend ('upward', 'downward', 'sideways')
            start_date: Trend start date
            end_date: Trend end date (optional)
            strength: Trend strength (optional)
        """
        try:
            cursor = self.conn.cursor()
            
            sql = """
                INSERT INTO trends
                (ticker, start_date, end_date, trend_type, strength)
                VALUES (?, ?, ?, ?, ?)
            """
            
            values = (ticker, start_date, end_date, trend_type, strength)
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Inserted trend for {ticker}")
            
        except Exception as e:
            logger.error(f"Error inserting trend: {str(e)}")
            raise
    
    # =========================================================================
    # Alert Operations
    # =========================================================================
    
    def create_alert(
        self,
        ticker: str,
        alert_type: str,
        severity: str,
        message: str,
        threshold_value: Optional[float] = None,
        actual_value: Optional[float] = None
    ):
        """
        Create an alert
        
        Args:
            ticker: Stock ticker symbol
            alert_type: Type of alert
            severity: Alert severity ('low', 'medium', 'high')
            message: Alert message
            threshold_value: Threshold value (optional)
            actual_value: Actual value (optional)
        """
        try:
            cursor = self.conn.cursor()
            
            sql = """
                INSERT INTO alerts
                (ticker, alert_type, alert_date, severity, message, 
                 threshold_value, actual_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                ticker,
                alert_type,
                datetime.now().strftime('%Y-%m-%d'),
                severity,
                message,
                threshold_value,
                actual_value
            )
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Created alert for {ticker}: {alert_type}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            raise
    
    def get_unread_alerts(self) -> List[Dict]:
        """
        Get all unread alerts
        
        Returns:
            List of alert dictionaries
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE is_read = 0 
                ORDER BY alert_date DESC, severity DESC
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []
    
    # =========================================================================
    # Report Operations
    # =========================================================================
    
    def save_report_metadata(
        self,
        report_name: str,
        report_type: str,
        tickers: List[str],
        file_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Save report metadata
        
        Args:
            report_name: Name of the report
            report_type: Type of report
            tickers: List of tickers analyzed
            file_path: Path to report file
            start_date: Analysis start date
            end_date: Analysis end date
        """
        try:
            cursor = self.conn.cursor()
            
            sql = """
                INSERT INTO reports
                (report_name, report_type, tickers, start_date, end_date, 
                 file_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                report_name,
                report_type,
                ','.join(tickers),
                start_date,
                end_date,
                file_path,
                'completed'
            )
            
            cursor.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Saved report metadata: {report_name}")
            
        except Exception as e:
            logger.error(f"Error saving report metadata: {str(e)}")
            raise
    
    # =========================================================================
    # Query Utilities
    # =========================================================================
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute custom SQL query
        
        Args:
            sql: SQL query string
            params: Query parameters (optional)
            
        Returns:
            DataFrame with query results
        """
        try:
            if params:
                df = pd.read_sql_query(sql, self.conn, params=params)
            else:
                df = pd.read_sql_query(sql, self.conn)
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()
    
    def get_latest_prices(self) -> pd.DataFrame:
        """Get latest prices using view"""
        return self.execute_query("SELECT * FROM latest_prices")


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager('test_analytics.db')
    
    with db:
        # Create tables
        db.create_tables()
    def create_tables(self):
        schema_path = 'schema_sql'
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        self.conn.executescript(schema_sql)
        self.conn.commit()
        print("Tables created successfully")
        # Insert company
        db.insert_company('WMT', 'Walmart Inc.', 
                         sector='Consumer Defensive',
                         industry='Discount Stores')
        
        # Get company
        company = db.get_company('WMT')
        print(f"Company: {company}")
        
        # Create alert
        db.create_alert('WMT', 'price_spike', 'high',
                       'Price increased by 5% in one day',
                       threshold_value=5.0, actual_value=5.2)
        
        # Get alerts
        alerts = db.get_unread_alerts()
        print(f"Alerts: {alerts}")
    
    print("Database operations completed successfully")
