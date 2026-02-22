"""
Live Sales Analytics Dashboard
Streamlit-based interactive dashboard for stock market analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .reportview-container .main .block-container {
        max-width: 1400px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Volatility (30-day rolling)
    df['Volatility'] = df['Daily_Return'].rolling(window=30).std()
    
    return df

def calculate_metrics(df):
    """Calculate key metrics"""
    if df.empty:
        return {}
    
    current_price = df['Close'].iloc[-1]
    start_price = df['Close'].iloc[0]
    price_change = current_price - start_price
    price_change_pct = (price_change / start_price) * 100
    
    avg_volume = df['Volume'].mean()
    total_volume = df['Volume'].sum()
    avg_price = df['Close'].mean()
    
    # Volatility
    daily_volatility = df['Close'].pct_change().std() * 100
    
    # High and Low
    high = df['High'].max()
    low = df['Low'].min()
    
    # RSI
    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 0
    
    return {
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'avg_volume': avg_volume,
        'total_volume': total_volume,
        'avg_price': avg_price,
        'daily_volatility': daily_volatility,
        'high': high,
        'low': low,
        'rsi': current_rsi
    }

def plot_price_chart(df, ticker):
    """Create interactive price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price & Moving Averages', 'MACD', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                  line=dict(color='gray', width=1, dash='dash'), opacity=0.5),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                  line=dict(color='gray', width=1, dash='dash'), opacity=0.5, 
                  fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )
    
    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red', width=1)),
        row=2, col=1
    )
    
    # MACD Histogram
    macd_histogram = df['MACD'] - df['Signal']
    colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
    fig.add_trace(
        go.Bar(x=df.index, y=macd_histogram, name='MACD Histogram', marker_color=colors, opacity=0.3),
        row=2, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
        row=3, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def plot_volume_chart(df, ticker):
    """Create volume chart"""
    fig = go.Figure()
    
    colors = ['green' if close >= open_price else 'red' 
              for close, open_price in zip(df['Close'], df['Open'])]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ))
    
    # Add moving average of volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_MA'],
        name='Volume MA (20)',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title=f'{ticker} Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_correlation_heatmap(tickers, period):
    """Create correlation heatmap for multiple stocks"""
    data = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            data[ticker] = df['Close']
    
    if not data:
        return None
    
    df_combined = pd.DataFrame(data)
    correlation = df_combined.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Stock Price Correlation Matrix',
        height=500
    )
    
    return fig

def plot_performance_comparison(tickers, period):
    """Compare performance of multiple stocks"""
    fig = go.Figure()
    
    for ticker in tickers:
        df = fetch_stock_data(ticker, period)
        if not df.empty:
            # Normalize to percentage change from start
            normalized = ((df['Close'] / df['Close'].iloc[0]) - 1) * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized,
                name=ticker,
                mode='lines'
            ))
    
    fig.update_layout(
        title='Normalized Performance Comparison (%)',
        xaxis_title='Date',
        yaxis_title='Return (%)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# ========================
# MAIN DASHBOARD
# ========================

def main():
    # Header
    st.title("📊 Live Sales Analytics Dashboard")
    st.markdown("Real-time stock market analysis with technical indicators")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    
    ticker_input = st.sidebar.text_input(
        "Enter Stock Ticker(s)",
        value='AAPL',
        help="Enter one or more tickers separated by commas (e.g., AAPL,MSFT,GOOGL)"
    )
    
    selected_tickers = [t.strip().upper() for t in ticker_input.split(',')]
    
    # Popular stocks selector
    st.sidebar.markdown("**Or select from popular stocks:**")
    additional_tickers = st.sidebar.multiselect(
        "Popular Stocks",
        default_tickers,
        default=[]
    )
    
    # Combine selections
    all_tickers = list(set(selected_tickers + additional_tickers))
    
    # Time period
    period_options = {
        '1 Day': '1d',
        '5 Days': '5d',
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y',
        'Max': 'max'
    }
    
    selected_period = st.sidebar.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=5  # Default to 1 Year
    )
    period = period_options[selected_period]
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will auto-refresh every 5 minutes")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    if not all_tickers:
        st.warning("Please enter at least one stock ticker")
        return
    
    # Tabs for different views
    if len(all_tickers) == 1:
        tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Technical Analysis", "📋 Details"])
        
        ticker = all_tickers[0]
        
        # Fetch data
        with st.spinner(f'Fetching data for {ticker}...'):
            df = fetch_stock_data(ticker, period)
        
        if df.empty:
            st.error(f"Failed to fetch data for {ticker}")
            return
        
        # Calculate indicators
        df = calculate_technical_indicators(df)
        metrics = calculate_metrics(df)
        
        # Tab 1: Overview
        with tab1:
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${metrics['current_price']:.2f}",
                    f"{metrics['price_change_pct']:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Day High",
                    f"${metrics['high']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Day Low",
                    f"${metrics['low']:.2f}"
                )
            
            with col4:
                st.metric(
                    "Avg Volume",
                    f"{metrics['avg_volume']/1e6:.2f}M"
                )
            
            with col5:
                rsi_delta = "Overbought" if metrics['rsi'] > 70 else "Oversold" if metrics['rsi'] < 30 else "Neutral"
                st.metric(
                    "RSI",
                    f"{metrics['rsi']:.1f}",
                    rsi_delta
                )
            
            # Price chart
            st.plotly_chart(plot_price_chart(df, ticker), use_container_width=True)
            
            # Volume chart
            st.plotly_chart(plot_volume_chart(df, ticker), use_container_width=True)
        
        # Tab 2: Technical Analysis
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Technical Indicators")
                
                # Current values
                st.markdown(f"""
                **Moving Averages:**
                - SMA 20: ${df['SMA_20'].iloc[-1]:.2f}
                - SMA 50: ${df['SMA_50'].iloc[-1]:.2f}
                
                **MACD:**
                - MACD: {df['MACD'].iloc[-1]:.2f}
                - Signal: {df['Signal'].iloc[-1]:.2f}
                
                **Bollinger Bands:**
                - Upper: ${df['BB_Upper'].iloc[-1]:.2f}
                - Middle: ${df['BB_Middle'].iloc[-1]:.2f}
                - Lower: ${df['BB_Lower'].iloc[-1]:.2f}
                """)
            
            with col2:
                st.subheader("Statistics")
                
                st.markdown(f"""
                **Price Statistics:**
                - Mean: ${metrics['avg_price']:.2f}
                - Std Dev: ${df['Close'].std():.2f}
                - Volatility: {metrics['daily_volatility']:.2f}%
                
                **Returns:**
                - Max Daily: {df['Daily_Return'].max():.2f}%
                - Min Daily: {df['Daily_Return'].min():.2f}%
                - Avg Daily: {df['Daily_Return'].mean():.2f}%
                """)
            
            # Distribution of returns
            st.subheader("Returns Distribution")
            fig = px.histogram(
                df.dropna(),
                x='Daily_Return',
                nbins=50,
                title='Daily Returns Distribution',
                labels={'Daily_Return': 'Daily Return (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Details
        with tab3:
            st.subheader("Recent Data")
            
            # Show last 20 rows
            display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
            available_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[available_cols].tail(20), use_container_width=True)
            
            # Download button
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv,
                file_name=f"{ticker}_data.csv",
                mime="text/csv"
            )
    
    else:
        # Multiple stocks view
        tab1, tab2, tab3 = st.tabs(["📊 Comparison", "🔗 Correlation", "📈 Individual"])
        
        # Tab 1: Comparison
        with tab1:
            st.subheader("Performance Comparison")
            
            # Metrics for all stocks
            cols = st.columns(len(all_tickers))
            for idx, ticker in enumerate(all_tickers):
                df = fetch_stock_data(ticker, period)
                if not df.empty:
                    df = calculate_technical_indicators(df)
                    metrics = calculate_metrics(df)
                    
                    with cols[idx]:
                        st.metric(
                            ticker,
                            f"${metrics['current_price']:.2f}",
                            f"{metrics['price_change_pct']:.2f}%"
                        )
            
            # Performance chart
            st.plotly_chart(plot_performance_comparison(all_tickers, period), use_container_width=True)
            
            # Summary table
            st.subheader("Summary Statistics")
            summary_data = []
            for ticker in all_tickers:
                df = fetch_stock_data(ticker, period)
                if not df.empty:
                    df = calculate_technical_indicators(df)
                    metrics = calculate_metrics(df)
                    summary_data.append({
                        'Ticker': ticker,
                        'Current Price': f"${metrics['current_price']:.2f}",
                        'Change %': f"{metrics['price_change_pct']:.2f}%",
                        'Volatility': f"{metrics['daily_volatility']:.2f}%",
                        'RSI': f"{metrics['rsi']:.1f}",
                        'Avg Volume': f"{metrics['avg_volume']/1e6:.2f}M"
                    })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Tab 2: Correlation
        with tab2:
            st.subheader("Correlation Analysis")
            corr_fig = plot_correlation_heatmap(all_tickers, period)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        
        # Tab 3: Individual
        with tab3:
            selected_stock = st.selectbox("Select Stock", all_tickers)
            
            df = fetch_stock_data(selected_stock, period)
            if not df.empty:
                df = calculate_technical_indicators(df)
                metrics = calculate_metrics(df)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"${metrics['current_price']:.2f}", f"{metrics['price_change_pct']:.2f}%")
                with col2:
                    st.metric("Volatility", f"{metrics['daily_volatility']:.2f}%")
                with col3:
                    st.metric("RSI", f"{metrics['rsi']:.1f}")
                with col4:
                    st.metric("Volume", f"{metrics['avg_volume']/1e6:.2f}M")
                
                # Charts
                st.plotly_chart(plot_price_chart(df, selected_stock), use_container_width=True)
                st.plotly_chart(plot_volume_chart(df, selected_stock), use_container_width=True)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption("Data provided by Yahoo Finance")

if __name__ == "__main__":
    main()
