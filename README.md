# 📊 Sales Analytics Dashboard

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

A comprehensive business intelligence platform for sales analytics using real-time financial data with SQL database integration and interactive dashboards.

**Author**: Suraj Goel  
**Email**: surajgoel501@gmail.com

---

## 🎯 Overview

This project provides end-to-end sales analytics capabilities including:

- Real-time data collection from Yahoo Finance API
- ETL pipeline with data validation and processing
- SQL database (SQLite/PostgreSQL/MySQL) for historical tracking
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Performance metrics (Sharpe ratio, volatility, drawdown)
- Interactive visualizations and HTML dashboards
- Automated alert system

## 📊 Features

### Data Collection
- Real-time financial data from Yahoo Finance
- Multiple tickers support
- Historical data retrieval (1mo to max)
- Company information extraction

### Data Processing
- Data cleaning and validation
- Technical indicators calculation
- Feature engineering
- Missing value handling

### Analytics
- Performance metrics
- Trend analysis
- Volatility assessment
- Risk metrics
- Signal generation

### Visualization
- Time series plots
- Candlestick charts
- Volume analysis
- Interactive dashboards
- HTML report generation

### Database
- SQLite/PostgreSQL/MySQL support
- Tables: companies, stock_data, technical_indicators, performance_metrics, analysis_results
- Query builders
- Data export (CSV, JSON)

---

## 🛠️ Technology Stack

- **Python 3.8+** - Core language
- **pandas** - Data manipulation
- **yfinance** - Financial data API
- **matplotlib, seaborn, plotly** - Visualizations
- **SQLAlchemy** - Database ORM
- **Jupyter** - Interactive notebooks

---

## 📝 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [SQL Database Guide](SQL_DATABASE_GUIDE.md)
- [Project Summary](PROJECT_SUMMARY.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Suraj Goel**  
📧 surajgoel501@gmail.com  
🔗 [GitHub Profile](https://github.com/surajgoel2201)

---

## ⚠️ Disclaimer

This project is for educational purposes. Financial data and analysis should not be construed as investment advice. The project uses publicly available market data as a proxy for sales analysis.
