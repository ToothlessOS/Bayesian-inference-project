import akshare as ak
import pandas as pd

"""
dataloader.py
Load Chinese stock market (A) data from akshare and select the 'fluctuation' 'close' columns
Author: ToothlessOS
"""

class DataLoader:

    def __init__(self, symbol: str, start_date: str, end_date: str, local: bool = False, path: str = ""):
        """
        symbol(str): stock symbol (example: '000001')
        startdate(str): start date of query (example: '20230101')
        startdate(str): end date of query (example: '20230301')
        TODO:
        local(bool): Default to False to load data from akshare, set to True to load data from local file
        """
        if local == False:
            self.name = symbol
            self.stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")
        else:
            self.stock_zh_a_hist_df = self.load_raw(path)
    
    def get(self) -> pd.DataFrame:
        data = self.stock_zh_a_hist_df[['日期', '股票代码', '振幅', '涨跌幅']]
        return data

    def save_raw(self, path: str):
        self.stock_zh_a_hist_df.to_csv(path, index=True)

    def load_raw(self, path: str):
        self.stock_zh_a_hist_df = pd.read_csv(path)

    def reload(self, symbol: str, start_date: str, end_date: str):
        self.stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="")