import pandas_ta as ta
import datetime
import yfinance as yf
import pandas as pd
import os
import numpy as np

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot

df = pd.read_csv("example.csv")
train = data_split(df, '2009-03-12', '2018-12-31')
trade = data_split(df, '2019-01-01', '2019-12-31')
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + 6 * stock_dimension

env_kwargs = {
    "hmax": 1,
    "initial_amount": 1000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ["rsi", "obv", "adx", "log_return", "kc_low", "kc_up"],
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

turbulence_threshold = np.quantile(train.turbulence.values, 1)
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()

env_trade, obs_trade = e_trade_gym.get_sb_env()

print("==============Model Training===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

model_kwargs = {"batch_size": 256,
                'buffer_size': 200000,
                'learning_rate': 0.00002}
agent = DRLAgent(env=env_train)
model_td3 = agent.get_model("td3", model_kwargs=model_kwargs)
model_td3 = agent.train_model(model_td3, tb_log_name="TD3_{}".format(now), total_timesteps=1000)

df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_td3,
                                                       test_data=trade,
                                                       test_env=env_trade,
                                                       test_obs=obs_trade)

print("==============Get Backtest Results===========")
perf_stats_all = BackTestStats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

print("==============Compare to DJIA===========")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
BackTestPlot(df_account_value,
             baseline_ticker = '^DJI',
             baseline_start = '2019-01-01',
             baseline_end = '2020-12-01')


