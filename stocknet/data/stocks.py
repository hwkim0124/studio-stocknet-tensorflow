
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, glob
import csv


# Using FinanceDataReader
# https://github.com/FinanceData/FinanceDataReader/wiki/Users-Guide


KOSPI_200_STOCKS_FILE = 'KOSPI200.csv'
STOCK_PRICE_START_DATE = '1998-01-01'
STOCK_PRICE_REFER_DATE = '2018-01-01'
STOCK_PRICE_DATA_DIR = './prices'


class Stock(): 
    def __init__(self, symbol, name, data=None): 
        self.symbol = symbol 
        self.name = name 
        self.data = data 
        self.prices = None 
        self.examples = [] 
        self.examples_mean = []
        self.examples_stde = [] 

    def label(self):
        return '{} ({})'.format(self.symbol, self.name)

    def recent_examples(self, days):
        data = self.examples[-days:]
        data = np.asarray(data).astype(np.float32)
        return data 



def read_kospi200_symbols(append_ks200=False):
    if not os.path.exists(KOSPI_200_STOCKS_FILE):
        print("Can't find KOSPI-200 stocks symbol file!")
        return

    stocks = []
    with open(KOSPI_200_STOCKS_FILE, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            print(line)
            stocks.append(Stock(line[0], line[1]))
            # stocks.append(tuple(line))
            break

        # KOSPI 200 index prices are available since 2001-01-02.
        stocks.append(Stock('KS200', 'KOSPI200지수'))

    print('%d stocks symbol loaded' % len(stocks))
    return stocks 


def read_stock_data(symbol, refer_date):
    df = fdr.DataReader(symbol, refer_date)
    df2 = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = df2.to_numpy()
    date = df.index.tolist()
    return data 


def read_kospi200_stock_data():
    stocks = read_kospi200_symbols()

    for stock in stocks:
        print('Fetching %s ... ' % stock.label(), end='')
        data = read_stock_data(stock.symbol, STOCK_PRICE_REFER_DATE)
        stock.data = data 
        print('%d trading days loaded' % data.shape[0])

    return stocks 


def download_kospi200_stock_prices():
    stocks = read_kospi200_symbols()

    if not os.path.exists(STOCK_PRICE_DATA_DIR):
        os.makedirs(STOCK_PRICE_DATA_DIR)

    count = 0
    for stock in stocks:
        symbol = stock[0]
        name = stock[1]

        # New string formatter
        # https://pyformat.info/
        path = '{}/{}_{}.csv'.format(STOCK_PRICE_DATA_DIR, symbol, name)
        df = fdr.DataReader(symbol, STOCK_PRICE_START_DATE)
        df.to_csv(path, encoding='ms949')
        print('%d ... %s (%s) price data exported' % (count, symbol, name))
        count += 1
        # break

    print('%d stocks price exported' % count)


def load_stock_prices(path=STOCK_PRICE_DATA_DIR):
    pattern = '{}/*.csv'.format(path)
    matched = glob.glob(pattern)
    print('%d stock price files found in \'%s\' (\'%s\')' % (len(matched), path, os.getcwd()))

    stocks = []
    for file in matched: 
        # if "001520" not in file: 
        #   continue 

        print('Reading %s ... ' % file, end='')

        # To read korean named file, use python parser engine. 
        df = pd.read_csv(file, engine='python')
        df2 = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = df2.to_numpy()

        print('%d trading days loaded' % data.shape[0])
        base = os.path.basename(file)
        name = os.path.splitext(base)[0]
        stocks.append(Stock(name[:6], name[7:], data))
        # break 
    
    print('Totally, %d stocks price data loaded' % len(stocks))
    return stocks 


if __name__ == '__main__':
    download_kospi200_stock_prices()
