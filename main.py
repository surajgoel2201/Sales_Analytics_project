"""
Main Execution Script
Runs the complete sales analytics pipeline
"""

import os
import sys
from datetime import datetime
import logging

# Import config first
try:
    import config
    print(config.__file__)
except ImportError:
    print("Warning: config module not found, using defaults")
    # Create a minimal config if it doesn't exist
    class config:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        LOG_FILE = 'logs/pipeline.log'
        DATA_DIR = 'data'
        RAW_DATA_DIR = 'data/raw'
        PROCESSED_DATA_DIR = 'data/processed'
        REPORTS_DIR = 'reports'
        FIGURES_DIR = 'figures'
        DEFAULT_PERIOD = '1y'
        HTML_REPORT = 'reports/dashboard.html'
        COMPANIES = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.'
        }

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules with error handling
try:
    from src.data_collector import StockDataCollector
except ImportError:
    print("Warning: Could not import StockDataCollector")
    StockDataCollector = None

try:
    from src.data_processor import DataProcessor
except ImportError:
    print("Warning: Could not import DataProcessor")
    DataProcessor = None

try:
    from src.analyzer import SalesAnalyzer
except ImportError:
    print("Warning: Could not import SalesAnalyzer")
    SalesAnalyzer = None

try:
    from src.visualizer import DashboardGenerator
except ImportError:
    print("Warning: Could not import DashboardGenerator")
    DashboardGenerator = None

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Directory structure verified")


def main():
    """
    Main execution function
    """
    logger.info("="*50)
    logger.info("Starting Sales Analytics Pipeline")
    logger.info("="*50)
    
    # Create directory structure
    create_directories()
    
    # Initialize components with error checking
    logger.info("Initializing components...")
    
    if StockDataCollector is None:
        logger.error("StockDataCollector not available. Cannot proceed.")
        return
    
    collector = StockDataCollector()
    
    if DataProcessor is None:
        logger.error("DataProcessor not available. Cannot proceed.")
        return
    
    processor = DataProcessor()
    
    if DashboardGenerator is None:
        logger.warning("DashboardGenerator not available. Visualizations will be skipped.")
        visualizer = None
    else:
        visualizer = DashboardGenerator()
    
    # Configuration
    tickers = list(config.COMPANIES.keys())
    period = config.DEFAULT_PERIOD
    
    logger.info(f"Analyzing {len(tickers)} companies: {', '.join(tickers)}")
    logger.info(f"Period: {period}")
    
    # Step 1: Collect Data
    logger.info("\n" + "="*50)
    logger.info("Step 1: Collecting Live Data from Yahoo Finance")
    logger.info("="*50)
    
    data_dict = {}
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        try:
            data = collector.get_stock_data(ticker, period=period)
            
            if data is not None and not data.empty:
                # Save raw data
                collector.save_data(data, f'{ticker}_raw.csv', 'raw')
                data_dict[ticker] = data
                logger.info(f"✓ {ticker}: {len(data)} records collected")
            else:
                logger.warning(f"✗ {ticker}: No data retrieved")
        except Exception as e:
            logger.error(f"✗ {ticker}: Error collecting data - {str(e)}")
    
    if not data_dict:
        logger.error("No data collected. Exiting.")
        return
    
    # Step 2: Process Data
    logger.info("\n" + "="*50)
    logger.info("Step 2: Processing and Cleaning Data")
    logger.info("="*50)
    
    processed_dict = {}
    for ticker, data in data_dict.items():
        logger.info(f"Processing {ticker}...")
        
        try:
            # Clean data
            cleaned = processor.clean_data(data)
            
            # Add technical indicators
            with_indicators = processor.add_technical_indicators(cleaned)
            
            # Add growth metrics
            with_growth = processor.calculate_growth_metrics(with_indicators)
            
            # Detect trends (FIXED: Check if method exists)
            if hasattr(processor, 'detect_trends'):
                processed = processor.detect_trends(with_growth)
            else:
                logger.warning(f"detect_trends method not found in DataProcessor, skipping trend detection")
                processed = with_growth
            
            # Save processed data
            processor.export_processed_data(processed, f'{ticker}_processed.csv')
            
            processed_dict[ticker] = processed
            logger.info(f"✓ {ticker}: Processing complete")
            
        except Exception as e:
            logger.error(f"✗ {ticker}: Error processing data - {str(e)}")
            logger.exception(e)
    
    if not processed_dict:
        logger.error("No data processed. Exiting.")
        return
    
    # Step 3: Analyze Data
    logger.info("\n" + "="*50)
    logger.info("Step 3: Analyzing Performance Metrics")
    logger.info("="*50)
    
    metrics_dict = {}
    insights_dict = {}
    
    for ticker, data in processed_dict.items():
        logger.info(f"Analyzing {ticker}...")
        
        try:
            if SalesAnalyzer is None:
                logger.warning("SalesAnalyzer not available, skipping analysis")
                continue
                
            analyzer = SalesAnalyzer(data)
            
            # Calculate metrics
            metrics = analyzer.calculate_key_metrics()
            metrics_dict[ticker] = metrics
            
            # Generate insights
            insights = analyzer.generate_insights()
            insights_dict[ticker] = insights
            
            # Print summary
            logger.info(f"\n{ticker} Summary:")
            logger.info(f"  Current Price: ${metrics.get('current_price', 0):.2f}")
            logger.info(f"  Price Change: {metrics.get('price_change_pct', 0):.2f}%")
            logger.info(f"  Avg Volume: {metrics.get('avg_volume', 0):,.0f}")
            logger.info(f"  Volatility: {metrics.get('daily_volatility', 0):.2f}%")
            
            if insights:
                logger.info(f"  Key Insights:")
                for insight in insights[:3]:  # Show top 3
                    logger.info(f"    - {insight}")
                    
        except Exception as e:
            logger.error(f"✗ {ticker}: Error analyzing data - {str(e)}")
            logger.exception(e)
    
    # Step 4: Generate Visualizations
    logger.info("\n" + "="*50)
    logger.info("Step 4: Creating Visualizations")
    logger.info("="*50)
    
    if visualizer is None:
        logger.warning("Skipping visualizations - DashboardGenerator not available")
    else:
        try:
            # Individual stock charts
            for ticker, data in processed_dict.items():
                logger.info(f"Creating charts for {ticker}...")
                
                try:
                    # Price trend chart
                    if hasattr(visualizer, 'plot_price_trend'):
                        visualizer.plot_price_trend(
                            data, ticker, 
                            save_path=f'{config.FIGURES_DIR}/{ticker}_price_trend.png'
                        )
                    
                    # Volume analysis
                    if hasattr(visualizer, 'plot_volume_analysis'):
                        visualizer.plot_volume_analysis(
                            data, ticker,
                            save_path=f'{config.FIGURES_DIR}/{ticker}_volume.png'
                        )
                    
                    # Candlestick chart
                    if hasattr(visualizer, 'plot_candlestick'):
                        visualizer.plot_candlestick(
                            data, ticker,
                            save_path=f'{config.FIGURES_DIR}/{ticker}_candlestick.html'
                        )
                    
                    # Technical analysis
                    if hasattr(visualizer, 'create_technical_analysis_chart'):
                        visualizer.create_technical_analysis_chart(
                            data, ticker,
                            save_path=f'{config.FIGURES_DIR}/{ticker}_technical.html'
                        )
                        
                except Exception as e:
                    logger.error(f"Error creating charts for {ticker}: {str(e)}")
            
            # Comparison charts
            logger.info("Creating comparison charts...")
            
            try:
                if hasattr(visualizer, 'plot_multiple_stocks'):
                    visualizer.plot_multiple_stocks(
                        processed_dict,
                        save_path=f'{config.FIGURES_DIR}/stock_comparison.png'
                    )
                
                if hasattr(visualizer, 'plot_correlation_heatmap'):
                    visualizer.plot_correlation_heatmap(
                        processed_dict,
                        save_path=f'{config.FIGURES_DIR}/correlation_heatmap.png'
                    )
                
                # Comprehensive dashboard
                logger.info("Creating comprehensive dashboard...")
                if hasattr(visualizer, 'create_performance_dashboard'):
                    visualizer.create_performance_dashboard(
                        processed_dict,
                        metrics_dict,
                        save_path=config.HTML_REPORT
                    )
                
                logger.info("✓ All visualizations created successfully")
                
            except Exception as e:
                logger.error(f"Error creating comparison charts: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.exception(e)
    
    # Step 5: Generate Summary Report
    logger.info("\n" + "="*50)
    logger.info("Step 5: Generating Summary Report")
    logger.info("="*50)
    
    try:
        # Create text report
        report_path = f'{config.REPORTS_DIR}/analysis_summary.txt'
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SALES ANALYTICS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Period: {period}\n")
            f.write("="*70 + "\n\n")
            
            for ticker in tickers:
                if ticker not in metrics_dict:
                    continue
                
                f.write(f"\n{ticker} - {config.COMPANIES.get(ticker, ticker)}\n")
                f.write("-"*70 + "\n")
                
                metrics = metrics_dict[ticker]
                f.write(f"\nKEY METRICS:\n")
                f.write(f"  Current Price: ${metrics.get('current_price', 0):.2f}\n")
                f.write(f"  Average Price: ${metrics.get('avg_price', 0):.2f}\n")
                f.write(f"  Price Change: ${metrics.get('price_change', 0):.2f} ")
                f.write(f"({metrics.get('price_change_pct', 0):.2f}%)\n")
                f.write(f"  Total Volume: {metrics.get('total_volume', 0):,.0f}\n")
                f.write(f"  Average Volume: {metrics.get('avg_volume', 0):,.0f}\n")
                f.write(f"  Daily Volatility: {metrics.get('daily_volatility', 0):.2f}%\n")
                
                if ticker in insights_dict and insights_dict[ticker]:
                    f.write(f"\nINSIGHTS:\n")
                    for insight in insights_dict[ticker]:
                        f.write(f"  • {insight}\n")
                
                f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("FILES GENERATED:\n")
            f.write(f"  - HTML Dashboard: {config.HTML_REPORT}\n")
            f.write(f"  - Figures: {config.FIGURES_DIR}/\n")
            f.write(f"  - Processed Data: {config.PROCESSED_DATA_DIR}/\n")
            f.write("="*70 + "\n")
        
        logger.info(f"✓ Summary report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        logger.exception(e)
    
    # Final Summary
    logger.info("\n" + "="*50)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*50)
    logger.info(f"Analyzed {len(processed_dict)} companies")
    
    if visualizer is not None and os.path.exists(config.FIGURES_DIR):
        logger.info(f"Generated {len(os.listdir(config.FIGURES_DIR))} visualizations")
    
    logger.info(f"\nView the dashboard: {config.HTML_REPORT}")
    if 'report_path' in locals():
        logger.info(f"View summary report: {report_path}")
    logger.info("="*50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)