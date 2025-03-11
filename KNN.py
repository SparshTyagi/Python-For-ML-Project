import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

class KNN:
    @staticmethod
    def train_knn(X, y):
        """Trains and evaluates a K-Nearest Neighbors classifier with hyperparameter tuning."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_knn_model.pkl"
        
        # Scale features - important for KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check for an existing checkpoint
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint KNN model...")
            best_knn = joblib.load(checkpoint_path)
            if isinstance(best_knn, tuple):
                best_knn, scaler = best_knn
        else:
            # Define hyperparameter grid
            param_distributions = {
                'n_neighbors': list(range(1, 31, 2)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3],  # Only relevant for minkowski metric
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            
            # Initialize KNN model
            knn = KNeighborsClassifier()
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                knn, param_distributions, n_iter=20, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=42
            )
            print("Training KNN model...")
            random_search.fit(X_train_scaled, y_train)
            best_knn = random_search.best_estimator_
            
            # Save the model with scaler
            joblib.dump((best_knn, scaler), checkpoint_path)
            print(f"Best KNN model saved to {checkpoint_path}")
        
        # Make predictions
        y_pred = best_knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        # Print evaluation results
        if 'random_search' in locals() and hasattr(random_search, 'best_params_'):
            print("\nBest KNN Parameters:", random_search.best_params_)
        else:
            print("\nLoaded checkpoint KNN model.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")
        
        return (best_knn, scaler), y_test, y_pred, accuracy, precision, recall, f1
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/knn_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"KNN Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, X, y, save_path="results/knn_feature_importance.png"):
        """Plots and saves KNN feature importance using permutation importance."""
        if isinstance(model, tuple):
            knn_model, scaler = model
            X_scaled = scaler.transform(X)
        else:
            knn_model = model
            X_scaled = X
            
        # Use permutation importance to determine feature importance
        print("Calculating permutation importance for KNN (this may take a while)...")
        perm_importance = permutation_importance(knn_model, X_scaled, y, n_repeats=5, random_state=42)
        
        # Create feature importance dataframe
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (KNN - Permutation Importance)')
        plt.xlabel('Mean Decrease in Accuracy')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importance plot saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    print("Loading feature data...")
    df = pd.read_csv("extracted_features.csv")
    X = df.drop(columns=['pid', 'time', 'label'], errors='ignore')
    y = df['label']
    
    print("\n===== KNN Model Training =====")
    best_knn, knn_y_test, knn_y_pred, knn_accuracy, knn_precision, knn_recall, knn_f1 = KNN.train_knn(X, y)
    KNN.plot_confusion_matrix(knn_y_test, knn_y_pred, knn_accuracy)
    KNN.plot_feature_importance(best_knn, X, y)
    
    print("\nModel training, evaluation, and visualization complete!")