
import keras
import os
import sys


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is '':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import stocknet.bin  # noqa: F401
    __package__ = "stocknet.bin"

from .. import models
from ..models import cnnnet
from ..data.dataset import Dataset


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def main(args=None):
    ds = Dataset() 
    ds.load_dataset()

    net = models.backbone('cnnnet')
    #net.train_stocknet(ds, load_weights=True)
    #net.evaluate_stocknet(ds)
    net.predict_stocknet(ds, train_data=False, test_data=True)
    return


if __name__ == '__main__':
    main()