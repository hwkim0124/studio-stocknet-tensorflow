
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

    
from .stocks import Stock, load_stock_prices 


NUM_FEATURES = 9
INPUT_DATA_SIZE = 60 
OUTPUT_DATA_SIZE = 3 
EXAMPLE_DATA_SIZE = (INPUT_DATA_SIZE + OUTPUT_DATA_SIZE)


class Dataset():

    def __init__(self):
        self.x_train = None 
        self.y_train = None 
        self.x_test = None 
        self.y_test = None 

    def build_dataset(self, save=True):
        stocks = load_stock_prices()
        preprare_examples(stocks)
        normalize_examples2(stocks)
        dataset = split_train_test_examples(stocks)
        
        self.x_train = np.asarray(dataset[0]).astype(np.float32)
        self.y_train = np.asarray(dataset[1]).astype(np.float32)
        self.x_test = np.asarray(dataset[2]).astype(np.float32)
        self.y_test = np.asarray(dataset[3]).astype(np.float32)

        if save:
            self.save_dataset() 
        return 

    def dataset(self):
        dataset = {}
        dataset['x_train'] = self.x_train
        dataset['y_train'] = self.y_train 
        dataset['x_test'] = self.x_test 
        dataset['y_test'] = self.y_test 
        return dataset 

    def load_dataset(self):
        self.x_train = np.load('./data/x_train.npy')
        self.y_train = np.load('./data/y_train.npy')
        self.x_test = np.load('./data/x_test.npy')
        self.y_test = np.load('./data/y_test.npy')

        print("Dataset loaded, training examples: %d, test examples: %d" % 
                (len(self.x_train), len(self.x_test)))
        return  

    def save_dataset(self):
        np.save('./data/x_train.npy', self.x_train)
        np.save('./data/y_train.npy', self.y_train)
        np.save('./data/x_test.npy', self.x_test)
        np.save('./data/y_test.npy', self.y_test)

        print("Dataset saved, training examples: %d, test examples: %d" %  
                (len(self.x_train), len(self.x_test)))
        return

    def train_size(self):
        if self.x_train is not None:
            return len(self.x_train)
        return 0 

    def test_size(self):
        if self.x_test is not None:
            return len(self.x_test)
        return 0


def preprare_examples(stocks, output_data=True):
    dataset = {} 
    for stock in stocks:
        print('Preparing stock data ... %s' % stock.label())
        data = stock.data 
        
        # Remove trading days in which any share trading for the stock not happend. 
        vols = data[:, 4]
        trades = np.where((vols > 0) & (data[:, 0] > 0))[0]
        values = np.squeeze(data[trades, :]) 
        print('Records size: %d, trading days: %d' % (len(vols), trades.size))

        closes = values[:, 3]
        prices = values[:, :4]
        
        # Moving averages of the passed time period.
        for n in [5, 10, 20, 60, 120]: 
            moving_avgs = np.zeros_like(closes)
            for k in range(n, len(closes)): 
                moving_avgs[k] = np.mean(closes[k-n:k])    

            # kernel = np.ones(2*n+1, )/n 
            # kernel[n:] = 0 
            # moving_avgs = np.convolve(closes, kernel, mode='same')

            for k in range(n): 
                moving_avgs[k] = np.mean(closes[:k+1]) 

            # x = np.arange(closes.size) 
            # plt.plot(x, closes, 'r.')
            # plt.plot(x, moving_avgs, 'b-')
            # plt.show() 

            moving_avgs = np.expand_dims(moving_avgs, axis=1)
            prices = np.hstack((prices, moving_avgs))

        # (Open, High, Low, Close, MA5, MA10, MA20, MA60, MA120)
        print('Trading data features: %s' % str(prices.shape))
        stock.prices = prices 

    # Prepare input sequences matrix in ascending order of symbol and time. 
    for stock in stocks:
        prices = stock.prices 
        size = prices.shape[0] 
        data = []
        period = EXAMPLE_DATA_SIZE if output_data else INPUT_DATA_SIZE
        
        for n in range(size-period):
            window = prices[n:n+period]
            data.append(window)
        stock.examples = data 

    return stocks 


def normalize_examples(stocks):
    for stock in stocks:
        print("Normalizing examples ... %s" % stock.label())

        data = stock.examples
        feats = []
        means = []
        stdes = []
        for i, example in enumerate(data):
            assert (example.shape[0] == EXAMPLE_DATA_SIZE or example.shape[0] == INPUT_DATA_SIZE)
            assert example.shape[1] == NUM_FEATURES 

            vals = list(example[:INPUT_DATA_SIZE, 3])
            mean = np.mean(vals)
            stde = np.std(vals) + 1

            norm = np.zeros_like(example, dtype=np.float)
            norm[:, 0] = (example[:, 0] - mean) / stde 
            norm[:, 1] = (example[:, 1] - example[:, 0]) / stde 
            norm[:, 2] = (example[:, 2] - example[:, 0]) / stde 
            for k in range(3, 9):
                norm[:, k] = (example[:, k] - mean) / stde

            feats.append(norm)
            means.append(mean)
            stdes.append(stde)
        
            # print(example)
            # print(norm)

        # Note that the mean and stdev for the input sequence should be stored 
        # to restore the original price values. 
        stock.examples = feats 
        stock.examples_mean = means 
        stock.examples_stde = stdes 

    return stocks 


def normalize_examples2(stocks):
    for stock in stocks:
        print("Normalizing examples ... %s" % stock.label())

        data = stock.examples
        feats = []
        means = []
        stdes = []
        for i, example in enumerate(data):
            assert (example.shape[0] == EXAMPLE_DATA_SIZE or example.shape[0] == INPUT_DATA_SIZE)
            assert example.shape[1] == NUM_FEATURES 

            vals = list(example[:INPUT_DATA_SIZE, 3])
            mean = np.mean(vals)
            stde = np.std(vals) + 1

            size = example.shape[0]
            data = example 

            c_open = np.array([1.0] + [(t[0] / t[1]) for t in zip(data[1:size, 0], data[0:size-1, 3])])
            c_high = np.array([t[0] / t[1] for t in zip(data[:, 1], data[:, 0])])
            c_lows = np.array([t[0] / t[1] for t in zip(data[:, 2], data[:, 0])])
            c_close = np.array([data[0, 3]/data[0, 0]] + [t[0] / t[1] for t 
                                in zip(data[1:size, 3], data[0:size-1, 3])])

            c_avg05 = np.array([t[0] / t[1] for t in zip(data[:, 4], data[:, 0])])
            c_avg10 = np.array([t[0] / t[1] for t in zip(data[:, 5], data[:, 0])])
            c_avg20 = np.array([t[0] / t[1] for t in zip(data[:, 6], data[:, 0])])
            c_avg60 = np.array([t[0] / t[1] for t in zip(data[:, 7], data[:, 0])])
            c_avg120 = np.array([t[0] / t[1] for t in zip(data[:, 8], data[:, 0])])

            norm = np.column_stack((c_open, c_high, c_lows, c_close, 
                        c_avg05, c_avg10, c_avg20, c_avg60, c_avg120))
            norm = norm - 1.0 
            
            feats.append(norm)
            means.append(mean)
            stdes.append(stde)
        
            # print(example)
            # print(norm)

        # Note that the mean and stdev for the input sequence should be stored 
        # to restore the original price values. 
        stock.examples = feats 
        stock.examples_mean = means 
        stock.examples_stde = stdes 

    return stocks 


def normalize_data(datum):
    featset = {}
    for symbol, data in datum.items():
        print('Prepare features ... %s' % symbol)
        size = data.shape[0] 
        c_open = np.array([1.0] + [t[0] / t[1] for t in zip(data[1:size, 0], data[0:size-1, 3])])
        c_high = np.array([t[0] / t[1] for t in zip(data[:, 1], data[:, 0])])
        c_lows = np.array([t[0]/t[1] for t in zip(data[:, 2], data[:, 0])])
        c_close = np.array([data[0, 3]/data[0, 0]] + [t[0] / t[1] for t 
                            in zip(data[1:size, 3], data[0:size-1, 3])])

        c_avg05 = np.array([t[0] / t[1] for t in zip(data[:, 4], data[:, 3])])
        c_avg10 = np.array([t[0] / t[1] for t in zip(data[:, 5], data[:, 3])])
        c_avg20 = np.array([t[0] / t[1] for t in zip(data[:, 6], data[:, 3])])
        c_avg60 = np.array([t[0] / t[1] for t in zip(data[:, 7], data[:, 3])])
        c_avg120 = np.array([t[0] / t[1] for t in zip(data[:, 8], data[:, 3])])

        feats = np.column_stack((c_open, c_high, c_lows, c_close, 
                    c_avg05, c_avg10, c_avg20, c_avg60, c_avg120))
        
        datum[symbol] = feats 

    return datum 


def split_train_test_examples(stocks):
    x_data = []
    y_data = []    
    for stock in stocks:
        for feat in stock.examples:
            x_data.append(feat[:INPUT_DATA_SIZE])
            y_data.append(feat[INPUT_DATA_SIZE:])

    size = len(x_data)
    print("Dataset examples size: %d" % size)

    indice = np.arange(size)
    np.random.shuffle(indice)
    
    train_size = int(size * 0.8)
    x_train = [x_data[i] for i in indice[:train_size]]
    y_train = [y_data[i][:, 3] for i in indice[:train_size]]
    x_test = [x_data[i] for i in indice[train_size:]]
    y_test = [y_data[i][:, 3] for i in indice[train_size:]]

    print("Training set size: %d, test set size: %d" % (len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test 


def save_dataset(dataset):
    np.save('./data/x_train.npy', dataset[0])
    np.save('./data/y_train.npy', dataset[1])
    np.save('./data/x_test.npy', dataset[2])
    np.save('./data/y_test.npy', dataset[3])

    print("Dataset saved, training examples: %d, test examples: %d" %  
            (len(dataset[0]), len(dataset[2])))


def load_dataset():
    dataset = {}
    dataset['x_train'] = np.load('./data/x_train.npy')
    dataset['y_train'] = np.load('./data/y_train.npy')
    dataset['x_test'] = np.load('./data/x_test.npy')
    dataset['y_test'] = np.load('./data/y_test.npy')

    print("Dataset loaded, training examples: %d, test examples: %d" % 
            (len(dataset['x_train']), len(dataset['x_test'])))
    return dataset 


def test_load_stocks_data():
    stocks = load_stock_prices()
    dataset = preprare_examples(stocks)
    dataset = normalize_examples(dataset)
    dataset = split_train_test_data(dataset)
    save_dataset(dataset)
    # load_dataset() 


if __name__ == '__main__':
    test_load_stocks_data()
    