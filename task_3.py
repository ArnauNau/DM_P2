# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set plot style for better aesthetics
sns.set(style='whitegrid', palette='bright', context='notebook')

# 1. Load the Digits Dataset
digits = load_digits()
X = digits.data  # Feature matrix: shape (1797, 64)
y = digits.target  # Target vector: shape (1797,)

# Display basic information about the dataset
print("Digits Dataset Loaded")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}\n")

# 2. Split the Data: 70% Training, 30% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Data Split into Training and Testing Sets")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}\n")

# 3. Normalize the Data Using StandardScaler Based on Training Data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

print("Data Normalized Using StandardScaler\n")

# 4. Principal Component Analysis (PCA)
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)
explained_variance_pca = pca.explained_variance_ratio_
print(f'PCA Explained Variance by Component: {explained_variance_pca}\n')

# 5. Truncated Singular Value Decomposition (TruncatedSVD)
svd = TruncatedSVD(n_components=2, random_state=42)
X_train_svd = svd.fit_transform(X_train_normalized)
X_test_svd = svd.transform(X_test_normalized)
explained_variance_svd = svd.explained_variance_ratio_
print(f'TruncatedSVD Explained Variance by Component: {explained_variance_svd}\n')

# 6. Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_normalized, y_train)
X_test_lda = lda.transform(X_test_normalized)
# Note: LDA's explained_variance_ratio_ is not directly comparable to PCA/SVD
print("LDA Projection Completed\n")

# 7. Create DataFrames for Visualization
# PCA DataFrame
df_pca = pd.DataFrame(
    X_train_pca,
    columns=['Principal Component 1', 'Principal Component 2']
)
df_pca['Class'] = y_train

# TruncatedSVD DataFrame
df_svd = pd.DataFrame(
    X_train_svd,
    columns=['SVD Component 1', 'SVD Component 2']
)
df_svd['Class'] = y_train

# LDA DataFrame
df_lda = pd.DataFrame(
    X_train_lda,
    columns=['LDA Component 1', 'LDA Component 2']
)
df_lda['Class'] = y_train

# 8. Visualization Functions
def plot_scatter(df, x_col, y_col, title, ax):
    """
    Plots a scatter plot for the given DataFrame and axes.

    Parameters:
    - df: pandas DataFrame containing the data.
    - x_col: str, name of the column for x-axis.
    - y_col: str, name of the column for y-axis.
    - title: str, title of the plot.
    - ax: matplotlib Axes object to plot on.
    """
    sns.scatterplot(
        x=x_col,
        y=y_col,
        hue='Class',
        palette='bright',
        data=df,
        alpha=0.7,
        ax=ax,
        edgecolor='k',
        linewidth=0.5
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title='Digit Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

# 9. Individual Scatter Plots
plt.figure(figsize=(18, 5))

# PCA Scatter Plot
plt.subplot(1, 3, 1)
plot_scatter(df_pca, 'Principal Component 1', 'Principal Component 2',
             'PCA Projection of Digits Dataset', plt.gca())

# TruncatedSVD Scatter Plot
plt.subplot(1, 3, 2)
plot_scatter(df_svd, 'SVD Component 1', 'SVD Component 2',
             'TruncatedSVD Projection of Digits Dataset', plt.gca())

# LDA Scatter Plot
plt.subplot(1, 3, 3)
plot_scatter(df_lda, 'LDA Component 1', 'LDA Component 2',
             'LDA Projection of Digits Dataset', plt.gca())

plt.tight_layout()
plt.show()

# 10. Side-by-Side Comparison Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# PCA Scatter Plot
plot_scatter(df_pca, 'Principal Component 1', 'Principal Component 2',
             'PCA Projection', axes[0])

# TruncatedSVD Scatter Plot
plot_scatter(df_svd, 'SVD Component 1', 'SVD Component 2',
             'TruncatedSVD Projection', axes[1])

# LDA Scatter Plot
plot_scatter(df_lda, 'LDA Component 1', 'LDA Component 2',
             'LDA Projection', axes[2])

plt.tight_layout()
plt.show()

# 11. Verification: Ensure PCA and SVD Projections are Different
projections_equal = np.allclose(X_train_pca, X_train_svd)
print(f'Are PCA and TruncatedSVD projections equal? {projections_equal}\n')

# 12. Display First 5 Projections for Each Method
print("First 5 PCA Projections:\n", X_train_pca[:5], "\n")
print("First 5 TruncatedSVD Projections:\n", X_train_svd[:5], "\n")
print("First 5 LDA Projections:\n", X_train_lda[:5], "\n")