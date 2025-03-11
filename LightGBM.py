import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb

class LGBM:
    @staticmethod
    def train_lgbm(X, y):
        """Trains and evaluates a LightGBM classifier with hyperparameter tuning."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_lgbm_model.pkl"
        
        # Check for an existing checkpoint
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint LightGBM model...")
            best_lgbm = joblib.load(checkpoint_path)
        else:
            # Define hyperparameter grid
            param_distributions = {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [20, 31, 50, 100],
                'min_child_samples': [5, 10, 20, 50],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            }
            
            # Initialize LightGBM model
            lgbm = lgb.LGBMClassifier(random_state=42)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                lgbm, param_distributions, n_iter=20, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=42
            )
            print("Training LightGBM model...")
            random_search.fit(X_train, y_train)
            best_lgbm = random_search.best_estimator_
            
            # Save the model
            joblib.dump(best_lgbm, checkpoint_path)
            print(f"Best LightGBM model saved to {checkpoint_path}")
        
        # Make predictions
        y_pred = best_lgbm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        # Print evaluation results
        if 'random_search' in locals() and hasattr(random_search, 'best_params_'):
            print("\nBest LightGBM Parameters:", random_search.best_params_)
        else:
            print("\nLoaded checkpoint LightGBM model.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")
        
        return best_lgbm, y_test, y_pred, accuracy, precision, recall, f1
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/lgbm_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"LightGBM Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, X, save_path="results/lgbm_feature_importance.png"):
        """Plots and saves LightGBM feature importance."""
        importance = model.feature_importances_
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (LightGBM)')
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
    
    print("\n===== LightGBM Model Training =====")
    best_lgbm, lgbm_y_test, lgbm_y_pred, lgbm_accuracy, lgbm_precision, lgbm_recall, lgbm_f1 = LGBM.train_lgbm(X, y)
    LGBM.plot_confusion_matrix(lgbm_y_test, lgbm_y_pred, lgbm_accuracy)
    LGBM.plot_feature_importance(best_lgbm, X)
    
    print("\nModel training, evaluation, and visualization complete!")