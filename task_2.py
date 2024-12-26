import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
digits = load_digits()
X = digits.data  # Features: Shape (1797, 64)
y = digits.target  # Labels: Shape (1797,)

# Split the data: 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,  # 30% for testing
    random_state=42,  # Ensures reproducibility
    stratify=y  # Maintains class proportions
)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform
X_train_normalized = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_normalized = scaler.transform(X_test)

# Verify the normalization
print("Training data mean (first 10 features):", np.mean(X_train_normalized, axis=0)[:10])
print("Training data std dev (first 10 features):", np.std(X_train_normalized, axis=0)[:10])

print("\nShapes after normalization:")
print("X_train_normalized:", X_train_normalized.shape)
print("X_test_normalized:", X_test_normalized.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate mean and std dev for each feature
feature_means = np.mean(X_train_normalized, axis=0)
feature_stds = np.std(X_train_normalized, axis=0)

# Reshape to 8x8 grid
mean_grid = feature_means.reshape((8, 8))
std_grid = feature_stds.reshape((8, 8))

# Plot Mean Intensities Heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(mean_grid, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Mean Intensity Heatmap (All Classes)')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()

# Plot Standard Deviations Heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(std_grid, annot=True, fmt=".2f", cmap='viridis', cbar=True)
plt.title('Standard Deviation Heatmap (All Classes)')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()