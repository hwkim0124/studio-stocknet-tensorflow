
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from stocks import load_stock_prices 


NUM_FEATURES = 9
INPUT_DATA_SIZE = 30 
OUTPUT_DATA_SIZE = 3 
EXAMPLE_DATA_SIZE = (INPUT_DATA_SIZE + OUTPUT_DATA_SIZE)


def preprare_data(stocks):
    dataset = {} 
    for symbol, data in stocks.items():
        print('Preparing stock data ... %s' % symbol)
    
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
        stocks[symbol] = prices 

    # Prepare input sequences matrix in ascending order of symbol and time. 
    for symbol, prices in stocks.items():
        size = prices.shape[0] 
        data = []
        for n in range(size-EXAMPLE_DATA_SIZE):
            data.append(prices[n:n+EXAMPLE_DATA_SIZE])
        dataset[symbol] = data 

    return dataset 


def normalize_examples(dataset):

    featset = {} 
    for symbol, data in dataset.items():
        print("Normalizing examples ... " % symbol)
        feats = []
        for i, example in enumerate(data):
            assert example.shape[0] == EXAMPLE_DATA_SIZE 
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
        
            # print(example)
            # print(norm)

        # Note that the mean and stdev for the input sequence should be stored 
        # to restore the original price values. 
        featset[symbol] = feats 

    return featset 


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


def split_train_eval_data(featset):
    x_data = []
    y_data = []    
    for symbol, feats in featset.items():
        for feat in feats:
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

    dataset = {}
    dataset['x_train'] = x_train
    dataset['y_train'] = y_train 
    dataset['x_test'] = x_test 
    dataset['y_test'] = y_test 
    return dataset 


def save_dataset(dataset):
    np.save('x_train.npy', dataset['x_train'])
    np.save('y_train.npy', dataset['y_train'])
    np.save('x_test.npy', dataset['x_test'])
    np.save('y_test.npy', dataset['y_test'])

    print("Dataset saved, training examples: %d, test examples: %d" %  
            (len(dataset['x_train']), len(dataset['x_test'])))


def load_dataset():
    dataset = {}
    dataset['x_train'] = np.load('x_train.npy')
    dataset['y_train'] = np.load('y_train.npy')
    dataset['x_test'] = np.load('x_test.npy')
    dataset['y_test'] = np.load('y_test.npy')

    print("Dataset loaded, training examples: %d, test examples: %d" % 
            (len(dataset['x_train']), len(dataset['x_test'])))
    return dataset 


def test_load_stocks_data():
    stocks = load_stock_prices()
    dataset = preprare_data(stocks)
    dataset = normalize_examples(dataset)
    dataset = split_train_eval_data(dataset)
    save_dataset(dataset)
    # load_dataset() 


if __name__ == '__main__':
    test_load_stocks_data()
    