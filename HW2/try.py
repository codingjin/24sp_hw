import numpy as np
import pandas as pd


np.random.seed(43)

xx = [1, 2, 3]
np.random.shuffle(xx)
print(xx)
np.random.shuffle(xx)
print(xx)
exit()

a, b = 10, 100
print(a, b)
exit()


w0 = np.random.uniform(-0.01, 0.01, 19)
b0 = np.random.uniform(-0.01, 0.01)

print("w0=",w0)
print("b0=",b0)

DATA_DIR = 'data/'

df = pd.read_csv(DATA_DIR + '2.csv').to_numpy()


"""
df = pd.read_csv(DATA_DIR + '1.csv')
df0 = df.columns.to_numpy()
print(df0)
"""
df1 = pd.read_csv(DATA_DIR + '2.csv').to_numpy()
#print(df1)

#for row in df1:
#    print("row[0]=", row[0], " row[7]=", row[7])

#np.random.shuffle(df1)
#print(df1)

w = w0
b = b0
print("before training:")
mistakes = 0
for datarow in df1:
    if datarow[0] * (np.dot(datarow[1:], w) + b) <= 0:
        mistakes += 1

print("%d mistakes" % (mistakes) )
lr = 1
for epoch in range(50):
    #print("epoch=", epoch)
    np.random.shuffle(df1)
    for datarow in df1:
        if datarow[0] * (np.dot(datarow[1:],w) + b) <= 0:
            w = w + lr*datarow[0]*datarow[1:]
            b = b + lr*datarow[0]

print("after training:")
mistakes = 0
for datarow in df1:
    if datarow[0] * (np.dot(datarow[1:], w) + b) <= 0:
        mistakes += 1

print("%d mistakes" % (mistakes) )


