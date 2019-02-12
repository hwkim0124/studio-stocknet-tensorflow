
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
import csv


# Using FinanceDataReader
# https://github.com/FinanceData/FinanceDataReader/wiki/Users-Guide


KOSPI_200_STOCKS_FILE = 'KOSPI200.csv'
STOCK_PRICE_START_DATE = '1998-01-01'
STOCK_PRICE_DATA_DIR = './prices'


def download_kospi200_stock_prices():
    if not os.path.exists(KOSPI_200_STOCKS_FILE):
        print("Can't find KOSPI-200 stocks symbol file!")
        return

    stocks = []
    with open(KOSPI_200_STOCKS_FILE, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            print(line)
            stocks.append(tuple(line))

        # KOSPI 200 index prices are available since 2001-01-02.
        stocks.append(('KS200', 'KOSPI200지수'))

    print('%d stocks symbol loaded' % len(stocks))
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
    print('%d stock price files found in %s' % (len(matched), path))

    prices = {}
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
        prices[name] = data 
        # break 
    
    print('Totally, %d stocks price data loaded' % len(prices))
    return prices 


if __name__ == '__main__':
    download_kospi200_stock_prices()
