import streamlit as st
from strategy1 import Strategy
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Group 18 - Strategy 2")

# Define a function to find equal weighted portfolio value and returns
def calculate_equal_weighted_portfolio(stock_symbols, start_date, end_date):
    equal_weighted_portfolio = pd.DataFrame()

    # Fetch stock data and calculate log returns for each stock
    for stock in stock_symbols:
        stock_data = yf.download(stock, start=start_date, end=end_date)
        stock_data['Log Return'] = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1))
        stock_data['Normal Return'] = np.exp(stock_data['Log Return']) - 1
        equal_weighted_portfolio[stock] = stock_data['Normal Return']

    # Calculate average normal return for the equal-weighted portfolio
    equal_weighted_portfolio['Average Normal Return'] = equal_weighted_portfolio.mean(axis=1)

    # Calculate portfolio value based on initial cash and average normal returns
    equal_weighted_portfolio['Portfolio Value'] = (1 + equal_weighted_portfolio['Average Normal Return']).cumprod()

    # Calculate log return of equal weighted portfolio
    equal_weighted_portfolio['Portfolio Log Return'] = np.log(equal_weighted_portfolio['Portfolio Value'] / equal_weighted_portfolio['Portfolio Value'].shift(1))

    return equal_weighted_portfolio['Portfolio Value'], equal_weighted_portfolio['Average Normal Return']

# Define a function to plot equal weighted portfolio vs dynamic strategy portfolio value
def plot_portfolio_comparison(dynamic_portfolio, equal_weighted_portfolio, start_date, end_date):
    # plot benchmark portfolio vs strategy portfolio returns
    plt.figure(figsize=(12, 6))
    plt.plot(dynamic_portfolio, label='Strategized Portfolio', color='orange')
    plt.plot(equal_weighted_portfolio, label='Equal-Weighted Portfolio', color='blue')
    plt.title(f"Portfolio Comparison: Strategized vs Equal-Weighted ({start_date} to {end_date})")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid()
    plt.show()

def plot_log_return_histogram_comparison(dynamic_portfolio_log_returns, equal_weighted_portfolio_log_returns, start_date, end_date):
    # Plot histograms for log returns
    plt.figure(figsize=(14, 5))

    # Dynamic portfolio histogram
    plt.subplot(1, 2, 1)
    plt.hist(equal_weighted_portfolio_log_returns, bins=30, color='blue', alpha=0.7)
    plt.title(f"Equal-Weighted Portfolio Log Returns ({start_date} to {end_date})")
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.grid()

    # Equal-weighted portfolio histogram
    plt.subplot(1, 2, 2)
    plt.hist(dynamic_portfolio_log_returns, bins=30, color='green', alpha=0.7)
    plt.title(f"Strategized Portfolio Log Returns ({start_date} to {end_date})")
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.grid()
    plt.show()

def plot_portfolio_drawdown_comparison(dynamic_portfolio, equal_weighted_portfolio, start_date, end_date):
    # Calculate cumulative max for dynamic and equal-weighted portfolios
    dynamic_cummax = dynamic_portfolio.cummax()
    equal_weighted_cummax = equal_weighted_portfolio.cummax()

    # Create two subplots side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot dynamic portfolio drawdown in the first subplot
    ax[0].plot(dynamic_cummax, label='Dynamic Portfolio Drawdown', color='orange')
    ax[0].plot(dynamic_portfolio, label='Dynamic Portfolio', color='blue')
    ax[0].set_title(f"Dynamic Portfolio Drawdown ({start_date} to {end_date})")
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Drawdown')
    ax[0].legend()
    ax[0].grid(True)

    # Plot equal-weighted portfolio drawdown in the second subplot
    ax[1].plot(equal_weighted_cummax, label='Equal-Weighted Portfolio Drawdown', color='orange')
    ax[1].plot(equal_weighted_portfolio, label='Equal-Weighted Portfolio Drawdown', color='blue')
    ax[1].set_title(f"Equal-Weighted Portfolio Drawdown ({start_date} to {end_date})")
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Drawdown')
    ax[1].legend()
    ax[1].grid(True)

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

# Define a function to find the best stop loss for each stock
def find_best_stop_loss(stock, start_date, end_date, stop_loss_range):
    best_stop_loss = None
    best_portfolio_value = None
    best_strategy_return = None

    # Loop through different stop losses and track performance
    for stop_loss_pct in stop_loss_range:
        stock_strategy = Strategy(stock, start_date, end_date, stop_loss_pct, filtered_benchmark_data)
        stock_strategy.generate_signals_positions()  # Generate signals and position
        strategy_returns = stock_strategy.calculate_strategy_returns()[0]  # Calculate strategy returns

        final_value = strategy_returns.iloc[-1]  # Get final portfolio value

        if best_portfolio_value is None or final_value > best_portfolio_value:
            best_portfolio_value = final_value
            best_stop_loss = stop_loss_pct
            best_strategy_return = strategy_returns

    return best_stop_loss, best_strategy_return

# Run the strategy
stock_symbols = ["META", "AAPL", "AMZN", "NFLX", "GOOG"]
start_date = '2015-01-01'
end_date = '2020-01-01'
best_stop_loss_dict = {
    "META": 0.1,
    "AAPL": 0.075, 
    "AMZN": 0.0875, 
    "NFLX": 0.0625, 
    "GOOG": 0.075
}

# Step 1: Obtain benchmark returns
benchmark_data = yf.download('SPY', start=start_date, end=end_date)
filtered_benchmark_data = benchmark_data[['Close', 'Adj Close', 'Open']]
filtered_benchmark_data['Log Return'] = np.log(filtered_benchmark_data['Adj Close'] / filtered_benchmark_data['Adj Close'].shift(1))

# Step 2: Calculate equal-weighted portfolio returns
equal_weighted_portfolio_value, equal_weighted_portfolio_returns = calculate_equal_weighted_portfolio(stock_symbols, start_date, end_date)

# Step 3: Apply dynamic strategy for each stock and accumulate portfolio returns
portfolio_value = pd.DataFrame(index=equal_weighted_portfolio_returns.index)
portfolio_value['Dynamic Strategy Value'] = 0  # Initialize column to accumulate portfolio values
portfolio_returns = pd.DataFrame(index=equal_weighted_portfolio_returns.index)
portfolio_returns['Dynamic Strategy Returns'] = 0  # Initialize column to accumulate portfolio returns
portfolio_returns['Equal Weighted Portfolio Returns'] = equal_weighted_portfolio_returns  # Initialize column to store equal weighted portfolio returns
portfolio_returns['Benchmark Returns (SPY)'] = filtered_benchmark_data['Log Return'].apply(np.exp) - 1 # Initialize column to store benchmark returns in portfolio returns df

overall_individual_stock_metrics = pd.DataFrame() # initialize dataframe to store individual stock and dynamic strategy for individual stock metrics

for stock in stock_symbols:
    # Find the best stop loss for the current stock
    best_stop_loss = best_stop_loss_dict[stock]
    # Initialize strategy with best stop loss for stock
    stock_strategy = Strategy(stock, start_date, end_date, best_stop_loss, filtered_benchmark_data)

    stock_strategy.generate_signals_positions()  # Generate signals and position
    dynamic_strategy_value, dynamic_strategy_returns = stock_strategy.calculate_strategy_returns() # store value and returns of individual stock
    portfolio_value[stock] = dynamic_strategy_value # store value
    portfolio_returns[stock] = dynamic_strategy_returns # store returns

    # Plot individual stock signals, returns, and histograms
    stock_strategy.plot_strategy_signals()
    # stock_strategy.visualise_returns()
    # stock_strategy.plot_histograms()
    # stock_strategy.plot_drawdown()
    # stock_metrics = stock_strategy.calculate_stock_metrics()
    # overall_individual_stock_metrics = pd.concat([overall_individual_stock_metrics, stock_metrics])

# Calculate the average value for the strategized portfolio
portfolio_value['Dynamic Strategy Value'] = portfolio_value[stock_symbols].mean(axis=1)
# Calculate the average returns for the strategized portfolio
for stock in stock_symbols:
    portfolio_returns[stock] = np.exp(portfolio_returns[stock]) - 1 # Convert log returns to normal
    portfolio_returns['Dynamic Strategy Returns'] = portfolio_returns[stock_symbols].mean(axis=1)  # Average of normal returns

# Step 4: Plot the comparison between dynamic strategized portfolio vs equal-weighted portfolio

# log returns forr histogram comparison
equal_weighted_portfolio_log_returns = np.log(equal_weighted_portfolio_value / equal_weighted_portfolio_value.shift(1))
dynamic_portfolio_log_returns = np.log(portfolio_value['Dynamic Strategy Value'] / portfolio_value['Dynamic Strategy Value'].shift(1))

plot_portfolio_comparison(portfolio_value['Dynamic Strategy Value'], equal_weighted_portfolio_value, start_date, end_date)
plot_log_return_histogram_comparison(dynamic_portfolio_log_returns, equal_weighted_portfolio_log_returns, start_date, end_date)
plot_portfolio_drawdown_comparison(portfolio_value['Dynamic Strategy Value'], equal_weighted_portfolio_value, start_date, end_date)

# Step 5: Print overall stock metrics
overall_individual_stock_metrics

