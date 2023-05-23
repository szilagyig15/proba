import numpy as np
import pandas as pd

def create_data():
    window_size = 100
    filepath = "VOO.csv"
    data = pd.read_csv(filepath,
        parse_dates=['Date'], index_col='Date')
    data['returns'] = np.log(data['Adj Close']).diff()
    data['squared_returns'] = data['returns'] ** 2
    cols = []
    for i in range(1, window_size + 1):
        col = f'lag_{i}'
        data[col] = data['squared_returns'].shift(i)
        cols.append(col)
    data.dropna(inplace=True)
    X = np.array(data[cols])
    y = np.array(data['squared_returns'])
    return X, y
create_data()

def load_data(self):
    self.data = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')


def calculate_returns(self):
    if self.return_type == 'continuous':
        self.data['returns'] = np.log(self.data['Adj Close']).diff()
        self.create_lagged_returns()  # Call this before squaring returns
    elif self.return_type == 'simple':
        self.data['returns'] = self.data['Adj Close'].pct_change()
        self.create_lagged_returns()  # Call this before squaring returns
    else:
        raise ValueError("Invalid return_type. Use 'continuous' or 'simple'")

def calculate_squared_returns(self):
    self.data['squared_returns'] = self.data['returns'] ** 2

def create_lagged_returns(self):
    for i in range(1, self.window_size + 1):
        self.data[f'lag_{i}'] = self.data['returns'].shift(i)
