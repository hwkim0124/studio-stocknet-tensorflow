import keras
import os
import sys


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is '':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import stocknet.bin  # noqa: F401
    __package__ = "stocknet.bin"

from .. import models
from ..data.stocks import read_kospi200_stock_data 
from ..data import dataset


def main(args=None):
    
    net = models.backbone('cnnnet')
    model = net.model_stocknet() 

    stocks = read_kospi200_stock_data()
    dataset.preprare_examples(stocks, output_data=False)
    dataset.normalize_examples(stocks)

    for stock in stocks:
        examples = stock.recent_examples(3)
        predicts = model.predict(examples)

        print('Predicting prices ... %s' % stock.label())
        print('estimate: ', predicts[0], ', last close: ', examples[0, 59, 3])
        print('estimate: ', predicts[1], ', last close: ', examples[1, 59, 3])
        print('estimate: ', predicts[2], ', last close: ', examples[2, 59, 3])


if __name__ == '__main__':
    main()