"""Stub for binance module when python-binance is not installed"""

class Client:
    KLINE_INTERVAL_1MINUTE = "1m"
    
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.api_key = api_key
        self.api_secret = api_secret
        
    def get_account(self):
        return {"status": "STUB"}
        
    def get_klines(self, symbol, interval, limit=500):
        # Return stub data
        import time
        ts = int(time.time() * 1000)
        return [[
            ts, "50000.0", "50100.0", "49900.0", "50050.0", 
            "100.0", ts + 60000, "5005000.0", 1000, 
            "50.0", "2502500.0", "0"
        ]]
    
    def get_historical_klines(self, symbol, interval, start_str):
        # Return stub historical data
        import time
        ts = int(time.time() * 1000)
        data = []
        for i in range(100):  # 100 candles
            ts_i = ts - (i * 60000)  # 1 minute apart
            data.append([
                ts_i, "50000.0", "50100.0", "49900.0", "50050.0", 
                "100.0", ts_i + 60000, "5005000.0", 1000, 
                "50.0", "2502500.0", "0"
            ])
        return data
    
    def get_all_tickers(self):
        return [
            {"symbol": "BTCUSDT", "price": "50000.0"},
            {"symbol": "ETHUSDT", "price": "3000.0"},
            {"symbol": "SOLUSDT", "price": "100.0"},
            {"symbol": "DOGEUSDT", "price": "0.1"}
        ]

class BinanceAPIException(Exception):
    pass