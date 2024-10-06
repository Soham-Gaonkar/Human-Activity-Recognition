import pickle
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tsfel


# Load datasets
X_train = np.load('Saved_Datasets/Raw_Acc_Data/X_train.npy')
X_test = np.load('Saved_Datasets/Raw_Acc_Data/X_test.npy')
y_train = np.load('Saved_Datasets/Raw_Acc_Data/y_train.npy')
y_test = np.load('Saved_Datasets/Raw_Acc_Data/y_test.npy')

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)



# Extract features using TSFEL
cfg = tsfel.get_features_by_domain()  # Get all features by default
X_train_features = tsfel.time_series_features_extractor(cfg, X_train, verbose=1, fs=50)
X_test_features = tsfel.time_series_features_extractor(cfg, X_test, verbose=1, fs=50)

# Remove highly correlated features
correlated_features = tsfel.correlated_features(X_train_features)
X_train_filtered = X_train_features.drop(correlated_features, axis=1)
X_test_filtered = X_test_features.drop(correlated_features, axis=1)

# Remove low variance features
variance_selector = VarianceThreshold(threshold=0)
X_train_reduced = variance_selector.fit_transform(X_train_filtered)
X_test_reduced = variance_selector.transform(X_test_filtered)

# Normalize features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_reduced)
X_test_normalized = scaler.transform(X_test_reduced)

# Apply PCA

pca = PCA(n_components=20)
X_train_pca_20 = pca.fit_transform(X_train_normalized)
X_test_pca_20 = pca.transform(X_test_normalized)

print("X_train_pca shape: ", X_train_pca_20.shape)
print("X_test_pca shape: ", X_test_pca_20.shape)


np.savez('Task-3 Prompt Engineering for Large Language Models (LLMs)/processed_data.npz',
         X_train_pca_20=X_train_pca_20,
         X_test_pca_20=X_test_pca_20,
         y_train=y_train,
         y_test=y_test)
print("Data saved successfully!")


print("PCA Explained Variance: ", pca.explained_variance_ratio_)
print("PCA Explained Variance Sum: ", sum(pca.explained_variance_ratio_))