import akshare as ak
import numpy as np
import pandas as pd
import datetime

"""
dataloader.py
Get daily closing data of Shanghai/Shenzhen Stock Exchange indexes and compute log returns. 
"""

def get(symbol: str = "sh000001", start_date: datetime.date = None, end_date: datetime.date = None):
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=symbol)
    
    # Get the required columns (date & close) + filter by date
    data = stock_zh_index_daily_df[["date", "close"]]
    
    if start_date is not None:
        data = data[data['date'] >= start_date]
    if end_date is not None:
        data = data[data['date'] <= end_date]
    
    # Compute daily log returns and add to a new column in dataframe
    data["log return"] = np.log(data["close"] / data["close"].shift(1))
    
    # Note: This will cause the first entry of "log return" to be NaN
    # Fix this by setting the first entry to 0
    data.loc[data.index[0], 'log return'] = 0

    # Write the data to a csv file under project root
    data.to_csv("data/output.csv")

    return data