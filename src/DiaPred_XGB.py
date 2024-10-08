import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load dataset from the same folder
file_path = "data/diabetes.csv"  # Ensure this file is in the same directory as your script
data = pd.read_csv(file_path)

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Split dataset into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size at each iteration
    'max_depth': [3, 4, 5],  # Maximum depth of the trees
    'subsample': [0.8, 0.9, 1.0]  # Fraction of samples used for training each tree
}
grid_search = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(np.unique(y))):
    for j in range(len(np.unique(y))):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
