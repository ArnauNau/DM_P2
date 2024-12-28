# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')

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

# 3. Define the Pipeline
# The pipeline consists of:
# - Preprocessing: StandardScaler, MinMaxScaler, or None
# - Dimensionality Reduction: PCA, TruncatedSVD, or LDA
# - Classifier: KNeighborsClassifier

# Initialize the Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Placeholder, will be set in GridSearchCV
    ('dim_reduction', PCA()),  # Placeholder, will be set in GridSearchCV
    ('knn', KNeighborsClassifier())  # KNN classifier
])

# 4. Define the Parameter Grid for GridSearchCV
param_grid = {
    # Preprocessing options
    'scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'],

    # Dimensionality Reduction options
    'dim_reduction': [PCA(), TruncatedSVD(), LinearDiscriminantAnalysis()],

    # Number of components for Dimensionality Reduction
    'dim_reduction__n_components': [2, 3, 5],

    # KNN hyperparameters
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],

    # KNN algorithm options
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# 5. Define the Cross-Validation Strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 6. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,  # Utilize all available cores
    verbose=2
)

# 7. Fit GridSearchCV on Training Data
print("Starting Grid Search...\n")
grid_search.fit(X_train, y_train)
print("\nGrid Search Completed!\n")

# 8. Display the Best Parameters and Best Score
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
print("Best Parameters:")
for param_name in sorted(param_grid.keys()):
    print(f"  {param_name}: {grid_search.best_params_[param_name]}")
print("\n")

# 9. Evaluate the Best Model on the Test Set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 11. Analyze GridSearchCV Results
# Convert cv_results_ to DataFrame for easier analysis
cv_results = pd.DataFrame(grid_search.cv_results_)

# Display top 5 results
print("Top 5 Grid Search Results:")
print(cv_results.sort_values(by='rank_test_score').head(5))

# 12. Visualize the Search Results
# For simplicity, we'll visualize how accuracy varies with number of neighbors for each dimensionality reduction technique

# Extract relevant data
results = cv_results[['param_dim_reduction', 'param_dim_reduction__n_components',
                      'param_knn__n_neighbors', 'param_knn__weights', 'mean_test_score']]


# Map dimensionality reduction to a string for easier plotting
def map_dim_reduction(row):
    if isinstance(row['param_dim_reduction'], PCA):
        return 'PCA'
    elif isinstance(row['param_dim_reduction'], TruncatedSVD):
        return 'TruncatedSVD'
    elif isinstance(row['param_dim_reduction'], LinearDiscriminantAnalysis):
        return 'LDA'
    else:
        return 'Unknown'


results['Dim_Reduction'] = results.apply(map_dim_reduction, axis=1)

# Plot accuracy vs. number of neighbors for each dimensionality reduction technique
plt.figure(figsize=(12, 8))
sns.lineplot(data=results, x='param_knn__n_neighbors', y='mean_test_score',
             hue='Dim_Reduction', marker='o')
plt.title('KNN Accuracy vs. Number of Neighbors for Different Dimensionality Reduction Techniques')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.legend(title='Dimensionality Reduction')
plt.show()

# Plot accuracy vs. number of components for each dimensionality reduction technique
plt.figure(figsize=(12, 8))
sns.lineplot(data=results, x='param_dim_reduction__n_components', y='mean_test_score',
             hue='Dim_Reduction', marker='o')
plt.title('Accuracy vs. Number of Components for Different Dimensionality Reduction Techniques')
plt.xlabel('Number of Components')
plt.ylabel('Cross-Validation Accuracy')
plt.legend(title='Dimensionality Reduction')
plt.show()

# 13. Feature Importance (Optional)
# For models that support feature importance (e.g., Random Forest), but KNN doesn't.
# Thus, we can skip this step or analyze component loadings.

# However, for LDA, we can analyze the class means.

if isinstance(grid_search.best_params_['dim_reduction'], LinearDiscriminantAnalysis):
    lda_best = grid_search.best_estimator_.named_steps['dim_reduction']
    print("Class Means in LDA Space:")
    print(lda_best.means_)