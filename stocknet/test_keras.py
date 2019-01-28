
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, LSTM
from keras.models import Model

a = Input(shape=(280, 256))

lstm = LSTM(32)
encoded = lstm(a)

print(lstm.output)