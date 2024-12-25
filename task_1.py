import numpy as np
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics

# Load the digits dataset
digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target

# Print the shape of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Print dataset description
print(digits.DESCR)

# Basic Statistics
print("\nBasic Statistics:")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

# Calculate mean and standard deviation for each feature
feature_means = np.mean(X, axis=0)
feature_stds = np.std(X, axis=0)

print("\nFeature-wise Mean (first 10 features):", feature_means[:10])
print("Feature-wise Std Dev (first 10 features):", feature_stds[:10])

# Count number of samples per class
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("\nNumber of samples per class:")
for cls, count in class_distribution.items():
    print(f"Class {cls}: {count}")

# Optional: Plotting some digits
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, digit, label in zip(axes.flatten(), X, y):
    ax.imshow(digit.reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
    if ax == axes.flatten()[9]:
        break
plt.tight_layout()
plt.show()


# Reshape mean intensities to 8x8 grid
mean_grid = feature_means.reshape((8, 8))

plt.figure(figsize=(6, 6))
plt.imshow(mean_grid, cmap='hot', interpolation='nearest')
plt.colorbar(label='Mean Intensity')
plt.title('Heatmap of Mean Intensities')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()


# Reshape standard deviations to 8x8 grid
std_grid = feature_stds.reshape((8, 8))

plt.figure(figsize=(6, 6))
plt.imshow(std_grid, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Standard Deviation')
plt.title('Heatmap of Standard Deviations')
plt.xlabel('Pixel Column')
plt.ylabel('Pixel Row')
plt.show()

# Identify unique classes
classes = np.unique(y)

# Initialize dictionaries to store mean and std per class
class_means = {}
class_stds = {}

for cls in classes:
    X_cls = X[y == cls]
    class_means[cls] = np.mean(X_cls, axis=0)
    class_stds[cls] = np.std(X_cls, axis=0)


# Function to reshape feature vectors to 8x8 grids
def reshape_to_grid(feature_vector):
    return feature_vector.reshape((8, 8))


# Number of classes
num_classes = len(classes)

# Create a figure with a grid of subplots: num_classes rows x 2 columns
fig, axes = plt.subplots(num_classes, 2, figsize=(10, 2 * num_classes), constrained_layout=True)
fig.suptitle('Mean and Std Dev Heatmaps per Digit Class', fontsize=16)

for idx, cls in enumerate(classes):
    # Mean Heatmap
    mean_grid = reshape_to_grid(class_means[cls])
    im_mean = axes[idx, 0].imshow(mean_grid, cmap='hot', interpolation='nearest')
    axes[idx, 0].set_title(f'Digit: {cls} - Mean')
    axes[idx, 0].axis('off')
    fig.colorbar(im_mean, ax=axes[idx, 0], fraction=0.046, pad=0.04)

    # Std Dev Heatmap
    std_grid = reshape_to_grid(class_stds[cls])
    im_std = axes[idx, 1].imshow(std_grid, cmap='coolwarm', interpolation='nearest')
    axes[idx, 1].set_title(f'Digit: {cls} - Std Dev')
    axes[idx, 1].axis('off')
    fig.colorbar(im_std, ax=axes[idx, 1], fraction=0.046, pad=0.04)

plt.show()