import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC

class SVM:
    @staticmethod
    def train_svm(X, y):
        """Trains and evaluates an SVM classifier with progress tracking and checkpointing."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_svm_model.pkl"

        # Check for an existing checkpoint
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint SVM model...")
            best_svm = joblib.load(checkpoint_path)
        else:
            # Define hyperparameter grid
            param_distributions = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
            # Initialize SVM model
            svm = SVC(random_state=42)
            # Perform randomized search with cross-validation and progress tracking
            random_search = RandomizedSearchCV(
                svm, param_distributions, n_iter=20, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1, random_state=42
            )
            with tqdm_joblib(tqdm(total=20, desc="SVM RandomizedSearchCV iterations")):
                random_search.fit(X_train, y_train)
            best_svm = random_search.best_estimator_
            joblib.dump(best_svm, checkpoint_path)
            print(f"Best SVM model saved to {checkpoint_path}")

        # Make predictions
        y_pred = best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)

        # Print evaluation results
        if 'random_search' in locals() and hasattr(random_search, 'best_params_'):
            print("\nBest SVM Parameters:", random_search.best_params_)
        else:
            print("\nLoaded checkpoint SVM model.")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")

        return best_svm, y_test, y_pred, accuracy, precision, recall, f1

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/svm_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"SVM Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")  # Save the figure before showing
        print(f"Confusion matrix plot saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_feature_importance(model, X, save_path="results/svm_feature_importance.png"):
        """Plots and saves SVM feature importance (available for linear kernel only)."""
        if model.kernel != 'linear':
            print("Feature importance visualization is only available for linear kernel SVM.")
            return
        importance = np.abs(model.coef_[0])
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (SVM)')
        plt.xlabel('Coefficient Magnitude')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")  # Save the figure before showing
        print(f"Feature importance plot saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    print("Loading feature data...")
    df = pd.read_csv("extracted_features.csv")
    # Drop non-relevant columns
    X = df.drop(columns=['pid', 'time', 'label'], errors='ignore')
    y = df['label']

    print("\n===== SVM Model Training =====")
    best_svm, svm_y_test, svm_y_pred, svm_accuracy, svm_precision, svm_recall, svm_f1 = SVM.train_svm(X, y)
    SVM.plot_confusion_matrix(svm_y_test, svm_y_pred, svm_accuracy)

    # Plot feature importance if using the linear kernel
    if best_svm.kernel == 'linear':
        SVM.plot_feature_importance(best_svm, X)

    joblib.dump(best_svm, "best_svm_model.pkl")
    print("Best SVM model saved to best_svm_model.pkl")
    print("\nModel training, evaluation, and visualization complete!")