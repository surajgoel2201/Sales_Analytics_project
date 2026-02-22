"""
Sales Analytics Package
"""

__version__ = '1.0.0'
__author__ = 'Suraj Goel'
__email__ = 'surajgoel501@gmail.com'

from .data_collector import StockDataCollector
from .data_processor import DataProcessor
from .analyzer import SalesAnalyzer
from .visualizer import DashboardGenerator
from .database import DatabaseManager

__all__ = [
    'StockDataCollector',
    'DataProcessor',
    'SalesAnalyzer',
    'DashboardGenerator',
    'DatabaseManager'
]
