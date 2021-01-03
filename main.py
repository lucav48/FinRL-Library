import datetime
import yfinance as yf
import pandas as pd
import os
import numpy as np

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot

full_data = pd.DataFrame()
for stock in config.DOW_30_TICKER[:10]:
    data = yf.download(tickers=stock,
                       interval="1d",
                       auto_adjust=True)
    data.dropna(how='any', axis=0, inplace=True)
    data["rsi"] = ta.rsi(data["Close"]) / 100
    data["obv"] = ta.obv(data["Close"], data["Volume"])
    data["obv"] = data["obv"] / data["obv"].max()
    data["adx"] = ta.adx(data["High"], data["Low"], data["Close"], length=14)["ADX_14"] / 100
    data["log_return"] = ta.log_return(data["Close"], length=14)
    kc = ta.kc(data["High"], data["Low"], data["Close"], length=14)
    data["kc_low"] = kc["KCL_14_2"]
    data["kc_up"] = kc["KCU_14_2"]
    data["kc_low"] = (data["kc_low"] - data["kc_low"].min()) / (data["kc_low"].max() - data["kc_low"].min())
    data["kc_up"] = (data["kc_up"] - data["kc_up"].min()) / (data["kc_up"].max() - data["kc_up"].min())
    data = data.loc[datetime.date(1999, 1, 1):]
    data = data.reset_index()
    data = data.drop("Volume", axis=1)
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Date": "date", "Volume": "volume"})
    data["tic"] = stock
    full_data = full_data.append(data)
full_data = full_data.sort_values(['date', 'tic'], ignore_index=True)

df = FeatureEngineer(use_technical_indicator=False,
                     tech_indicator_list=[],
                     use_turbulence=True,
                     user_defined_feature=False).preprocess_data(full_data.copy())
train = data_split(df, '2009-03-12', '2018-12-31')
trade = data_split(df, '2019-01-01', '2019-12-31')

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + 6 * stock_dimension
env_setup = EnvSetup(stock_dim=stock_dimension,
                     state_space=state_space,
                     hmax=1,
                     initial_amount=1000,
                     transaction_cost_pct=0.001,
                     tech_indicator_list=["rsi", "obv", "adx", "log_return", "kc_up", "kc_low"])
env_train = env_setup.create_env_training(data=train,
                                          env_class=StockEnvTrade)

agent = DRLAgent(env=env_train)
print("==============Model Training===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

model_kwargs = {"batch_size": 256,
                'buffer_size': 200000,
                'learning_rate': 0.00002}

model_td3 = agent.get_model("td3", model_kwargs=model_kwargs)
model_td3 = agent.train_model(model_td3, tb_log_name="TD3_{}".format(now), total_timesteps=100000)

turbulence_threshold = np.quantile(train.turbulence.values, 1)

env_trade, obs_trade = env_setup.create_env_trading(data=trade,
                                                    env_class=StockEnvTrade,
                                                    turbulence_threshold=turbulence_threshold)

df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_td3,
                                                       test_data=trade,
                                                       test_env=env_trade,
                                                       test_obs=obs_trade)

print("==============Get Backtest Results===========")
perf_stats_all = BackTestStats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
print(perf_stats_all)
print("==============Compare to DJIA===========")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
print(BackTestPlot(df_account_value,
                   baseline_ticker='^DJI',
                   baseline_start='2019-01-01',
                   baseline_end='2020-12-01'))
