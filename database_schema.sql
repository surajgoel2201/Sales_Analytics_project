-- ============================================================================
-- Sales Analytics Database Schema
-- Compatible with SQLite / PostgreSQL / MySQL
-- ============================================================================

-- ============================================================================
-- Table: companies
-- ============================================================================
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY,  -- SQLite: INTEGER PK auto-increments; PostgreSQL/MySQL: use SERIAL/AUTO_INCREMENT
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    employees INTEGER,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Table: stock_data
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_data (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10, 2),
    high_price DECIMAL(10, 2),
    low_price DECIMAL(10, 2),
    close_price DECIMAL(10, 2),
    adjusted_close DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

CREATE INDEX IF NOT EXISTS idx_stock_data_company_date ON stock_data(company_id, date);
CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date);

-- ============================================================================
-- Table: technical_indicators
-- ============================================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL,
    date DATE NOT NULL,
    sma_50 DECIMAL(10, 2),
    sma_200 DECIMAL(10, 2),
    ema_12 DECIMAL(10, 2),
    ema_26 DECIMAL(10, 2),
    rsi DECIMAL(5, 2),
    bollinger_upper DECIMAL(10, 2),
    bollinger_middle DECIMAL(10, 2),
    bollinger_lower DECIMAL(10, 2),
    macd DECIMAL(10, 4),
    macd_signal DECIMAL(10, 4),
    macd_histogram DECIMAL(10, 4),
    volatility DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

CREATE INDEX IF NOT EXISTS idx_indicators_company_date ON technical_indicators(company_id, date);

-- ============================================================================
-- Table: performance_metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL,
    date DATE NOT NULL,
    daily_return DECIMAL(8, 4),
    cumulative_return DECIMAL(10, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    volatility_30d DECIMAL(5, 4),
    volume_ma BIGINT,
    price_change_pct DECIMAL(8, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, date),
    FOREIGN KEY (company_id) REFERENCES companies(id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_company_date ON performance_metrics(company_id, date);

-- ============================================================================
-- Table: analysis_results
-- ============================================================================
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY,
    company_id INTEGER NOT NULL