import numpy as np
import pandas as pd
import os


X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep=r'\s+', header=None)
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', sep=r'\s+', header=None)

X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep=r'\s+', header=None)
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', sep=r'\s+', header=None)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


os.makedirs('Saved_Datasets/UCI_561', exist_ok=True)

np.save("Saved_Datasets/UCI_561/UCI_X_train.npy",X_train)
np.save("Saved_Datasets/UCI_561/UCI_X_test.npy",X_test)
np.save("Saved_Datasets/UCI_561/UCI_y_train.npy",y_train)
np.save("Saved_Datasets/UCI_561/UCI_y_test.npy",y_test)

print("Data saved successfully!")