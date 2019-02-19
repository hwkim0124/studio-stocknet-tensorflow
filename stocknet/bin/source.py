import keras
import os
import sys


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is '':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import stocknet.bin  # noqa: F401
    __package__ = "stocknet.bin"


from ..data.dataset import Dataset


def main(args=None):
    ds = Dataset() 
    ds.build_dataset() 
    return 


if __name__ == '__main__':
    main()