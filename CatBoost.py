import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from catboost import CatBoostClassifier, Pool

class CatBoost:
    @staticmethod
    def train_catboost(X, y):
        """Trains and evaluates a CatBoost classifier with hyperparameter tuning."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_catboost_model.pkl"
        
        # Check for an existing checkpoint
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint CatBoost model...")
            best_catboost = joblib.load(checkpoint_path)
        else:
            # Define hyperparameter grid
            param_distributions = {
                'iterations': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7],
                'random_strength': [0.1, 1, 10],
                'bagging_temperature': [0, 1, 10],
                'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
            }
            
            # Initialize CatBoost model with verbose=0 to reduce output
            catboost = CatBoostClassifier(random_seed=42, verbose=0)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                catboost, param_distributions, n_iter=15, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=42
            )
            print("Training CatBoost model...")
            random_search.fit(X_train, y_train, verbose=False)
            best_catboost = random_search.best_estimator_
            
            # Save the model
            joblib.dump(best_catboost, checkpoint_path)
            print(f"Best CatBoost model saved to {checkpoint_path}")
        
        # Make predictions
        y_pred = best_catboost.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        # Print evaluation results
        if 'random_search' in locals() and hasattr(random_search, 'best_params_'):
            print("\nBest CatBoost Parameters:", random_search.best_params_)
        else:
            print("\nLoaded checkpoint CatBoost model.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")
        
        return best_catboost, y_test, y_pred, accuracy, precision, recall, f1
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/catboost_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"CatBoost Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, X, save_path="results/catboost_feature_importance.png"):
        """Plots and saves CatBoost feature importance."""
        importance = model.get_feature_importance()
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (CatBoost)')
        plt.xlabel('Importance')
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
    
    print("\n===== CatBoost Model Training =====")
    best_catboost, catboost_y_test, catboost_y_pred, catboost_accuracy, catboost_precision, catboost_recall, catboost_f1 = CatBoost.train_catboost(X, y)
    CatBoost.plot_confusion_matrix(catboost_y_test, catboost_y_pred, catboost_accuracy)
    CatBoost.plot_feature_importance(best_catboost, X)
    
    print("\nModel training, evaluation, and visualization complete!")