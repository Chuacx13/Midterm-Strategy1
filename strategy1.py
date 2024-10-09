import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ta
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class Strategy:
    def __init__(self, stock_symbol, start_date, end_date, stop_loss_pct, benchmark_data):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.stop_loss_pct = stop_loss_pct
        self.stock_data = self.get_stock_data()
        self.benchmark_data = benchmark_data
        self.calculate_ema()
        self.calculate_rsi()
        self.calculate_sma()
        self.generate_mean_weighted_ema_diff()
        self.entry_price = None
        self.best_signal_col = None
        self.best_strat_return_col = None

    def get_stock_data(self):
        # Fetch historical stock data
        stock_data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        filtered_stock_data = stock_data[['Close', 'Adj Close', 'Open']]
        return filtered_stock_data

    def calculate_sma(self):
        self.stock_data['SMA50'] = SMAIndicator(close=self.stock_data['Adj Close'], window= 50, fillna= True).sma_indicator()
        self.stock_data['SMA200'] = SMAIndicator(close=self.stock_data['Adj Close'], window= 200, fillna= True).sma_indicator()

    def calculate_ema(self):
        # Calculate EMA5 and EMA10
        self.stock_data.loc[:, 'EMA5'] = ta.trend.EMAIndicator(close=self.stock_data['Adj Close'], window=5).ema_indicator()
        self.stock_data.loc[:, 'EMA10'] = ta.trend.EMAIndicator(close=self.stock_data['Adj Close'], window=10).ema_indicator()
        self.stock_data.loc[:, 'EMA_diff'] = self.stock_data['EMA5'] - self.stock_data['EMA10']

    def calculate_rsi(self):
        # Calculate RSI
        rsi5 = RSIIndicator(close=self.stock_data['Adj Close'], window=5).rsi()
        self.stock_data['RSI5'] = rsi5

    def generate_mean_weighted_ema_diff(self):
        """
        This function applies an exponentially weighted moving average to the EMA difference,
        giving more importance to recent data points.
        """
        # Calculate the mean EMA diff
        self.stock_data['Weighted_EMA_diff'] = self.stock_data['EMA_diff'].ewm(span=5, adjust=False).mean()

        return self.stock_data['Weighted_EMA_diff']

    def generate_signals_positions(self):
        quantile_combinations = [(0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.1, 0.9), (0.15, 0.85)] # try different rsi quantiles
        position_cols = []

        for q_low, q_high in quantile_combinations:
          # Instantiate columns to store each quantile variation's returns and signals
          signal_col = f'Signal_{q_low}_{q_high}'
          position_col = f'Position_{q_low}_{q_high}'
          self.stock_data[signal_col] = 0  # Initialize signal column

          # Instantiate buy cooldown, sell cooldown and entry price for short term trading strategy
          buy_cooldown = 0
          sell_cooldown = 0
          self.entry_price = None

          # Generate default position using long term trend in the case where short term trading strategy is not executed
          self.stock_data[signal_col] = 0
          self.stock_data[signal_col] = np.where((self.stock_data['Adj Close'] < self.stock_data['SMA50']) & (self.stock_data['Adj Close'] < self.stock_data['SMA200']), -1, self.stock_data[signal_col])  # default short position
          self.stock_data[signal_col] = np.where((self.stock_data['Adj Close'] >= self.stock_data['SMA50']) | (self.stock_data['Adj Close'] >= self.stock_data['SMA200']), 1, self.stock_data[signal_col])  # default long position

          for i in range(len(self.stock_data)):
              if i == len(self.stock_data) - 1: # exit loop once last line is reached
                  break

              stop_loss = None # checker to check whether stop loss is executed

              # handle stop loss for long position
              if self.entry_price is not None and self.stock_data.iloc[i]['Adj Close'] <= self.entry_price * (1 - self.stop_loss_pct) and self.stock_data.iloc[i-1][signal_col] == 1:
                  self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = -1 # if stop loss is executed for long position, instantly go into short position
                  self.entry_price = None  # Reset entry price after triggering stop-loss
                  sell_cooldown = 5 # hold short position for 5 days unless stop loss executed
                  stop_loss = True # stop loss executed

              # Handle stop-loss for short position
              elif self.entry_price is not None and self.stock_data.iloc[i]['Adj Close'] >= self.entry_price * (1 + self.stop_loss_pct) and self.stock_data.iloc[i-1][signal_col] == -1:
                  self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = 1 # if stop loss is executed for long position, instantly go into long position
                  self.entry_price = None  # Reset entry price after triggering stop-loss
                  buy_cooldown = 5 # hold long position for 5 days unless stop loss executed
                  stop_loss = True # stop loss executed

              if sell_cooldown == 0 and buy_cooldown == 0 and stop_loss is not True: # do not run if stop loss executed
                  # Evaluate sell_condition and buy_condition for the current row
                  rsi_upper = self.stock_data['RSI5'].rolling(window=20).quantile(q_high).iloc[i] # find upper quantile of RSI from past 20 days
                  rsi_lower = self.stock_data['RSI5'].rolling(window=20).quantile(q_low).iloc[i] # find lower quantile of RSI from past 20 days
                  sell_condition = (self.stock_data.iloc[i]['Weighted_EMA_diff'] < self.stock_data.iloc[i-1]['Weighted_EMA_diff']) and (self.stock_data.iloc[i]['Weighted_EMA_diff'] < self.stock_data.iloc[i-2]['Weighted_EMA_diff']) and (self.stock_data.iloc[i]['RSI5'] > rsi_upper) and (self.stock_data.iloc[i][signal_col] == 1)
                  buy_condition = (self.stock_data.iloc[i]['Weighted_EMA_diff'] < self.stock_data.iloc[i-1]['Weighted_EMA_diff']) and (self.stock_data.iloc[i]['Weighted_EMA_diff'] < self.stock_data.iloc[i-2]['Weighted_EMA_diff']) and (self.stock_data.iloc[i]['RSI5'] < rsi_lower) and (self.stock_data.iloc[i][signal_col] == -1)

                  if sell_condition: # execute sell condition
                      self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = -1
                      self.entry_price = self.stock_data.iloc[i + 1]['Open'] # entry price is open price on the next day after signal
                      sell_cooldown = 5

                  elif buy_condition: # execute buy condition
                      self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = 1
                      self.entry_price = self.stock_data.iloc[i + 1]['Open'] # entry price is open price on the next day after signal
                      buy_cooldown = 5

                  elif self.stock_data.iloc[i][signal_col] != self.stock_data.iloc[i-1][signal_col]: # check if default position changed if buy/sell condition not executed
                      self.entry_price = self.stock_data.iloc[i + 1]['Open']  # Short/Long position entry price on the next day after signal
                  else:
                      continue
              else:
                  if buy_cooldown > 0 and stop_loss is not True: # in long position currently and stop loss not executed
                      buy_cooldown -= 1
                      self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = 1

                  if sell_cooldown > 0 and stop_loss is not True: # in short position currently and stop loss not executed
                      sell_cooldown -= 1
                      self.stock_data.iloc[i, self.stock_data.columns.get_loc(signal_col)] = -1

          self.stock_data[position_col] = self.stock_data[signal_col].shift(1) # generate position
          self.stock_data[signal_col] = np.where(self.stock_data[signal_col] != self.stock_data[position_col], self.stock_data[signal_col], 0) # removed duplicate signals
          position_cols.append(position_col)
        return position_cols


    def calculate_strategy_returns(self):
        quantile_combinations = [(0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.1, 0.9), (0.15, 0.85)]
        portfolio_value_cols = []

        for q_low, q_high in quantile_combinations: # calculate returns for each quantile combination
          position_col = f'Position_{q_low}_{q_high}'
          strategy_return_col = f'Strategy_Return_{q_low}_{q_high}'
          portfolio_value_col = f'Portfolio_Value_{q_low}_{q_high}'
          # Calculate returns based on buy/sell signals
          self.stock_data['Log Return'] = np.log(self.stock_data['Adj Close'] / self.stock_data['Adj Close'].shift(1))
          self.stock_data[strategy_return_col] = self.stock_data['Log Return'] * self.stock_data[position_col]

          # Convert strategy log returns to normal returns
          normal_returns = np.exp(self.stock_data[strategy_return_col]) - 1

          # Calculate portfolio value using cumulative product of normal returns
          self.stock_data[portfolio_value_col] = (1 + normal_returns).cumprod()
          portfolio_value_cols.append(portfolio_value_col)

        final_values = self.stock_data[portfolio_value_cols].iloc[-1]  # Get the final portfolio values
        best_portfolio_col = final_values.idxmax()  # Find the column with the maximum final value
        _, q_low, q_high = best_portfolio_col.split('_')[1:]  # Split and extract quantile values (ignoring "Portfolio" and "Value")

        # Obtain the corresponding signal and strategy return column name using the quantile combination
        self.best_signal_col = f'Signal_{q_low}_{q_high}'
        self.best_strat_return_col = f'Strategy_Return_{q_low}_{q_high}'

        return self.stock_data[best_portfolio_col], self.stock_data[self.best_strat_return_col] # return value of best performing quantile combination for stock its log returns

    def plot_strategy_signals(self):
        # Create subplots with a shared x-axis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                'Price with SMA and Signals',
                'RSI Indicator',
                'EMA Difference'
            ),
            vertical_spacing=0.1
        )

        # Plot Close Price with SMA lines and signals in the first subplot
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['Adj Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black'),
            hoverinfo='x+y',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['SMA50'],
            mode='lines',
            name='SMA50',
            line=dict(color='yellow', dash='dash'),
            hoverinfo='x+y',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['SMA200'],
            mode='lines',
            name='SMA200',
            line=dict(color='orange', dash='dash'),
            hoverinfo='x+y',
        ), row=1, col=1)

        # Plot buy signals
        buy_signals = self.stock_data[self.stock_data[self.best_signal_col] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Adj Close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy Signal',
            hoverinfo='x+y',
        ), row=1, col=1)

        # Plot sell signals
        sell_signals = self.stock_data[self.stock_data[self.best_signal_col] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Adj Close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell Signal',
            hoverinfo='x+y',
        ), row=1, col=1)

        # RSI Plot in the second subplot
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data["RSI5"],
            mode='lines',
            name='RSI5',
            line=dict(color='purple'),
            hoverinfo='x+y',
        ), row=2, col=1)

        # EMA Difference Plot in the third subplot
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['EMA_diff'],
            mode='lines',
            name='EMA5 - EMA10 Difference',
            line=dict(color='blue'),
            hoverinfo='x+y',
        ), row=3, col=1)

        # Update layout to include a range slider only on the bottom plot without a chart in it
        fig.update_layout(
            height=900,
            title=f'{self.stock_symbol} - Trading Strategy with Signals, RSI, and EMA',
            xaxis3=dict(  # This applies the range slider to the x-axis of the third plot only
                title='Date',
                rangeslider=dict(
                    visible=True,
                    bgcolor='lightgrey',  # Slider background color to distinguish it from the plot area
                    thickness=0.05  # Slider thickness for a cleaner look
                ),
                type='date'
            ),
            legend=dict(
                x=1.05,  # Position the legend outside the plot area to the right
                y=1,
                traceorder='normal',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='x unified',
            template='plotly_white',
        )

        # Display the interactive plot in Streamlit
        st.plotly_chart(fig)


    def plot_drawdown(self):
        self.stock_data["Gross_Cum_Returns"] = self.stock_data[self.best_strat_return_col].cumsum().apply(np.exp)
        self.stock_data["Cum_Max"] = self.stock_data["Gross_Cum_Returns"].cummax()
        plt.figure(figsize=(15, 6))
        plt.plot(self.stock_data[["Gross_Cum_Returns", "Cum_Max"]].dropna())
        plt.title(f"Drawdown for {self.stock_symbol} at {self.stop_loss_pct} stop loss")
        plt.grid()
        plt.show()


    # Define a function to calculate performance of individual stock vs strategy
    def calculate_stock_metrics(self, risk_free_rate=0.00):
        """
        Calculate stock metrics including Alpha, Beta, Standard Deviation,
        Annualized Return, Sharpe Ratio, Maximum Drawdown, Sortino Ratio, Information Ratio, Calmar Ratio and Treynor Ratio.

        Assume risk free rate to be 0
        """
        # Daily normal returns of the stock
        stock_returns = self.stock_data['Log Return'].apply(np.exp) - 1
        strategy_returns = self.stock_data[self.best_strat_return_col].apply(np.exp) - 1
        benchmark_returns = self.benchmark_data['Log Return'].apply(np.exp) - 1

        # Remove NaN values
        merged_data = pd.DataFrame({
            'stock_returns': stock_returns,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns
        }).dropna()

        # Calculate Beta
        covariance_stock = np.cov(merged_data['stock_returns'], merged_data['benchmark_returns'])[0, 1]
        variance_benchmark = np.var(merged_data['benchmark_returns'])
        beta_stock = covariance_stock / variance_benchmark

        covariance_strat = np.cov(merged_data['strategy_returns'], merged_data['benchmark_returns'])[0, 1]
        beta_strat = covariance_strat / variance_benchmark

        # Calculate Alpha
        stock_yearly_return = merged_data['stock_returns'].mean() * 252
        strat_yearly_return = merged_data['strategy_returns'].mean() * 252
        benchmark_yearly_return = merged_data['benchmark_returns'].mean() * 252
        alpha_stock = (stock_yearly_return - risk_free_rate - beta_stock * (benchmark_yearly_return - risk_free_rate))
        alpha_strat = (strat_yearly_return - risk_free_rate - beta_strat * (benchmark_yearly_return - risk_free_rate))

        # Calculate Standard Deviation
        stock_daily_sd = merged_data['stock_returns'].std()
        strat_daily_sd = merged_data['strategy_returns'].std()
        std_dev_strat = strat_daily_sd * np.sqrt(252)  # Annualized standard deviation for Strategy
        std_dev_stock = stock_daily_sd * np.sqrt(252)  # Annualized standard deviation for Stock

        # Calculate Sharpe Ratio
        daily_SR_stock = (merged_data['stock_returns'].mean() - risk_free_rate) / stock_daily_sd
        daily_SR_strat = (merged_data['strategy_returns'].mean() - risk_free_rate) / strat_daily_sd
        annual_SR_stock = daily_SR_stock * np.sqrt(252)
        annual_SR_strat = daily_SR_strat * np.sqrt(252)

        # Calculate Maximum Drawdown
        cumulative_returns_strat = (1 + merged_data['strategy_returns']).cumprod()
        cum_max_strat = cumulative_returns_strat.cummax()
        drawdown_strat = cum_max_strat - cumulative_returns_strat
        max_drawdown_strat = drawdown_strat.max()

        cumulative_returns_stock = (1 + merged_data['stock_returns']).cumprod()
        cum_max_stock = cumulative_returns_stock.cummax()
        drawdown_stock = cum_max_stock - cumulative_returns_stock
        max_drawdown_stock = drawdown_stock.max()

        # Calculate Sortino Ratio
        converted_returns_strat = merged_data['strategy_returns'].apply(lambda x: 0 if x > 0 else x)
        squared_converted_returns_strat = converted_returns_strat ** 2
        squared_sum_converted_returns_strat = np.sum(squared_converted_returns_strat)
        downside_deviation_strat = np.sqrt(squared_sum_converted_returns_strat / len(converted_returns_strat))
        annual_converted_sd_strat = downside_deviation_strat * np.sqrt(252)
        sortino_ratio_strat = (strat_yearly_return) / annual_converted_sd_strat

        converted_returns_stock = merged_data['stock_returns'].apply(lambda x: 0 if x > 0 else x)
        squared_converted_returns_stock = converted_returns_stock ** 2
        squared_sum_converted_returns_stock = np.sum(squared_converted_returns_stock)
        downside_deviation_stock = np.sqrt(squared_sum_converted_returns_stock / len(converted_returns_stock))
        annual_converted_sd_stock = downside_deviation_stock * np.sqrt(252)
        sortino_ratio_stock = (stock_yearly_return) / annual_converted_sd_stock

        # Calculte Tracking Error
        tracking_error_strat = np.std(merged_data['strategy_returns'] - merged_data['benchmark_returns']) * np.sqrt(252)
        tracking_error_stock = np.std(merged_data['stock_returns'] - merged_data['benchmark_returns']) * np.sqrt(252)

        # Calculate Information Ratio
        information_ratio_strat = (strat_yearly_return - benchmark_yearly_return) / tracking_error_strat
        information_ratio_stock = (stock_yearly_return - benchmark_yearly_return) / tracking_error_stock

        # Calculate Calmar Ratio
        calmar_ratio_strat = strat_yearly_return / max_drawdown_strat
        calmar_ratio_stock = stock_yearly_return / max_drawdown_stock

        # Calculate Treynor Ratio
        treynor_ratio_strat = (strat_yearly_return - risk_free_rate) / beta_strat
        treynor_ratio_stock = (stock_yearly_return - risk_free_rate) / beta_stock


        # Assuming self.stock_symbol is your stock identifier
        metrics_df = pd.DataFrame({
            'Stock Returns': [stock_yearly_return],
            'Strat Returns': [strat_yearly_return],
            'Stock Alpha': [alpha_stock],
            'Strat Alpha': [alpha_strat],
            'Stock Beta': [beta_stock],
            'Strat Beta': [beta_strat],
            'Stock SD': [std_dev_stock],
            'Strat SD': [std_dev_strat],
            'Stock SR': [annual_SR_stock],
            'Strat SR': [annual_SR_strat],
            'Max Drawdown Stock': [max_drawdown_stock],
            'Max Drawdown Strat': [max_drawdown_strat],
            'Sortino Ratio Stock': [sortino_ratio_stock],
            'Sortino Ratio Strat': [sortino_ratio_strat],
            'Information Ratio Stock': [information_ratio_stock],
            'Information Ratio Strat': [information_ratio_strat],
            'Calmar Ratio Stock': [calmar_ratio_stock],
            'Calmar Ratio Strat': [calmar_ratio_strat],
            'Treynor Ratio Stock': [treynor_ratio_stock],
            'Treynor Ratio Strat': [treynor_ratio_strat]
        })

        # Set the stock as the row header (index)
        metrics_df.index = [self.stock_symbol]

        return metrics_df
