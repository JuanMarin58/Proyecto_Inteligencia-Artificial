import pandas as pd
from app.model.arquitectura import Model_LSTM

model = Model_LSTM(
    variables=["^GSPC", "^DJI", "^IXIC", "^RUT", "PL=F", "UUP", "GC=F", "^GDAXI", "SMCI", "BBD", "XRT", "XLK", "XLF", "SI=F", "AAPL", "GOOG", "AMZN", "MSFT", "TSLA", "NVDA"],
    target_variable='AAPL',
    start_date='2025-04-01',
    look_back=30,
    future_periods=3
)

data = model.load_data()
variables = data[['GOOG', 'AAPL', 'AMZN', '^GSPC']].copy()

print(variables.tail(5))

