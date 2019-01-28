
import FinanceDataReader as fdr
import matplotlib.pyplot as plt


df = fdr.DataReader('KS11', '1992-01-01')
print(df.shape)

print(df.tail())
df['Close'].plot()
plt.show()