import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LR:
    @staticmethod
    def train_lr(X, y):
        """Trains and evaluates a Logistic Regression classifier with hyperparameter tuning."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_lr_model.pkl"
        
        # Scale features - important for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check for an existing checkpoint
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint LR model...")
            best_lr = joblib.load(checkpoint_path)
        else:
            # Define hyperparameter grid with valid combinations
            param_distributions = [
                {
                    'penalty': ['l2'],
                    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None]
                },
                {
                    'penalty': ['l1'],
                    'solver': ['liblinear', 'saga'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None]
                },
                {
                    'penalty': ['elasticnet'],
                    'solver': ['saga'],
                    'l1_ratio': [0.5],  # you can try other values if desired
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None]
                },
                {
                    'penalty': [None],
                    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'class_weight': ['balanced', None]
                }
            ]
            
            # Initialize LR model
            lr = LogisticRegression(random_state=42, max_iter=1000)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                lr, param_distributions, n_iter=20, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=42
            )
            print("Training Logistic Regression model...")
            random_search.fit(X_train_scaled, y_train)
            best_lr = random_search.best_estimator_
            
            # Save the model
            joblib.dump((best_lr, scaler), checkpoint_path)
            print(f"Best LR model saved to {checkpoint_path}")
        
        if isinstance(best_lr, tuple):
            best_lr, scaler = best_lr
            X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = best_lr.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        # Print evaluation results
        if 'random_search' in locals() and hasattr(random_search, 'best_params_'):
            print("\nBest LR Parameters:", random_search.best_params_)
        else:
            print("\nLoaded checkpoint LR model.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")
        
        return (best_lr, scaler), y_test, y_pred, accuracy, precision, recall, f1
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/lr_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Logistic Regression Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(model, X, save_path="results/lr_feature_importance.png"):
        """Plots and saves Logistic Regression feature importance based on coefficients."""
        if isinstance(model, tuple):
            model = model[0]  # Extract LR model from tuple with scaler
            
        importance = np.abs(model.coef_[0])
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (Logistic Regression)')
        plt.xlabel('Coefficient Magnitude')
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
    
    print("\n===== Logistic Regression Model Training =====")
    best_lr, lr_y_test, lr_y_pred, lr_accuracy, lr_precision, lr_recall, lr_f1 = LR.train_lr(X, y)
    LR.plot_confusion_matrix(lr_y_test, lr_y_pred, lr_accuracy)
    LR.plot_feature_importance(best_lr, X)
    
    print("\nModel training, evaluation, and visualization complete!")