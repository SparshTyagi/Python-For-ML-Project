import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

class SVM:
    @staticmethod
    def train_svm(X, y):
        """Trains and evaluates an SVM classifier with progress tracking."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter grid
        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }

        # Initialize SVM model
        svm = SVC(random_state=42)

        # Perform randomized search with cross-validation and progress tracking
        random_search = RandomizedSearchCV(svm, param_distributions, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, verbose=0, random_state=42)
        
        # Wrap random_search.fit call with tqdm_joblib to track progress
        with tqdm_joblib(tqdm(total=20, desc="SVM RandomizedSearchCV iterations")):
            random_search.fit(X_train, y_train)

        # Get the best model
        best_svm = random_search.best_estimator_

        # Make predictions
        y_pred = best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)

        # Print evaluation results
        print("\nBest SVM Parameters:", random_search.best_params_)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")

        return best_svm, y_test, y_pred, accuracy, precision, recall, f1
    # @staticmethod
    # def train_svm(X, y):
    #     """Trains and evaluates an SVM classifier with progress tracking."""
    #     # Split data into training and testing sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #     # Define hyperparameter grid
    #     param_grid = {
    #         'C': [0.1, 1, 10, 100],
    #         'kernel': ['linear', 'rbf', 'poly'],
    #         'gamma': ['scale', 'auto', 0.1, 0.01]
    #     }

    #     # Calculate total iterations for progress tracking
    #     total_iter = len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma']) * 5

    #     # Initialize SVM model
    #     svm = SVC(random_state=42)

    #     # Perform grid search with cross-validation and progress tracking
    #     grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        
    #     # Wrap grid_search.fit call with tqdm_joblib to track progress
    #     with tqdm_joblib(tqdm(total=total_iter, desc="SVM GridSearchCV iterations")):
    #         grid_search.fit(X_train, y_train)

    #     # Get the best model
    #     best_svm = grid_search.best_estimator_

    #     # Make predictions
    #     y_pred = best_svm.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #     sober_accuracy = tn / (tn + fp)

    #     # Print evaluation results
    #     print("\nBest SVM Parameters:", grid_search.best_params_)
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1 Score: {f1:.4f}")
    #     print(f"Sober Accuracy: {sober_accuracy:.4f}")

    #     return best_svm, y_test, y_pred, accuracy, precision, recall, f1

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/svm_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"SVM Confusion Matrix (Accuracy: {accuracy:.4f})")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_feature_importance(model, X, save_path="results/svm_feature_importance.png"):
        """Plots and saves SVM feature importance (for linear kernel only)."""
        # Check if model has a linear kernel
        if model.kernel != 'linear':
            print("Feature importance visualization is only available for linear kernel SVM.")
            return
            
        # Get feature importance from coefficients
        importance = np.abs(model.coef_[0])
        features = X.columns
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Features (SVM)')
        plt.xlabel('Coefficient Magnitude')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


class MLP:
    @staticmethod
    def train_mlp(X, y):
        """Trains and evaluates an MLP classifier with progress tracking."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        # Calculate total iterations for progress tracking
        total_iter = (len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * 
                      len(param_grid['solver']) * len(param_grid['alpha']) * 
                      len(param_grid['learning_rate']) * 5)

        # Initialize MLP Classifier
        mlp = MLPClassifier(max_iter=500, random_state=42)

        # Perform grid search with cross-validation and progress tracking
        grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
        
        # Wrap grid_search.fit call with tqdm_joblib to track progress
        with tqdm_joblib(tqdm(total=total_iter, desc="MLP GridSearchCV iterations")):
            grid_search.fit(X_train, y_train)

        # Get the best model
        best_mlp = grid_search.best_estimator_

        # Make predictions
        y_pred = best_mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)

        # Print evaluation results
        print("\nBest MLP Parameters:", grid_search.best_params_)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")

        return best_mlp, y_test, y_pred, accuracy, precision, recall, f1

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/mlp_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"MLP Confusion Matrix (Accuracy: {accuracy:.4f})")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_feature_importance(model, X, y, save_path="results/mlp_feature_importance.png"):
        """Plots and saves MLP feature importance using permutation importance."""
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Features (MLP)')
        plt.xlabel('Permutation Importance')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

def compare_model_performance(rf_metrics, svm_metrics, mlp_metrics, save_path="results/model_comparison.png"):
    """Compares and visualizes the performance metrics of different models."""
    models = ['Random Forest', 'SVM', 'MLP']
    metrics = {
        'Accuracy': [rf_metrics[0], svm_metrics[0], mlp_metrics[0]],
        'Precision': [rf_metrics[1], svm_metrics[1], mlp_metrics[1]],
        'Recall': [rf_metrics[2], svm_metrics[2], mlp_metrics[2]],
        'F1 Score': [rf_metrics[3], svm_metrics[3], mlp_metrics[3]]
    }
    
    df = pd.DataFrame(metrics, index=models)
    
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', ylim=(0, 1), rot=0)
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_combined_confusion_matrices(svm_cm, mlp_cm, rf_cm=None, save_path="results/combined_confusion_matrices.png"):
    """Plots and saves combined confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3 if rf_cm is not None else 2, figsize=(15, 5))
    
    # Plot SVM confusion matrix
    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
    axes[0].set_title("SVM")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # Plot MLP confusion matrix
    sns.heatmap(mlp_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
    axes[1].set_title("MLP")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("")
    
    # Plot RF confusion matrix if provided
    if rf_cm is not None:
        sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        axes[2].set_title("Random Forest")
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    print("Loading feature data...")
    df = pd.read_csv("extracted_features.csv")
    
    # Drop non-relevant columns
    X = df.drop(columns=['pid', 'time', 'label'], errors='ignore')
    y = df['label']

    print("\n===== MLP Model Training =====")
    best_mlp, mlp_y_test, mlp_y_pred, mlp_accuracy, mlp_precision, mlp_recall, mlp_f1 = MLP.train_mlp(X, y)
    mlp_cm = confusion_matrix(mlp_y_test, mlp_y_pred)
    MLP.plot_confusion_matrix(mlp_y_test, mlp_y_pred, mlp_accuracy)
    
    # Plot MLP feature importance
    print("\nCalculating MLP feature importance...")
    MLP.plot_feature_importance(best_mlp, X, y)

    print("\n===== SVM Model Training =====")
    best_svm, svm_y_test, svm_y_pred, svm_accuracy, svm_precision, svm_recall, svm_f1 = SVM.train_svm(X, y)
    svm_cm = confusion_matrix(svm_y_test, svm_y_pred)
    SVM.plot_confusion_matrix(svm_y_test, svm_y_pred, svm_accuracy)
    
    # Plot SVM feature importance if linear kernel
    if best_svm.kernel == 'linear':
        SVM.plot_feature_importance(best_svm, X)
    

    
    
    # Get RandomForest results - assuming this has been run separately
    # We'll use placeholder values here - replace with actual values from your RandomForest run
    print("\n===== Comparing Model Performance =====")
    # For demonstration - you should replace these with your actual RF metrics
    rf_accuracy = 0.835    # Example value - replace with actual
    rf_precision = 0.691  # Example value - replace with actual 
    rf_recall = 0.585      # Example value - replace with actual
    rf_f1 = 0.634         # Example value - replace with actual
    
    # Create a mock RF confusion matrix for visualization
    # Replace this with actual RF confusion matrix if available
    rf_cm = np.array([[5386, 494], [785, 1108]])  # Example values
    
    # Compare model performance with bar charts
    compare_model_performance(
        [rf_accuracy, rf_precision, rf_recall, rf_f1],
        [svm_accuracy, svm_precision, svm_recall, svm_f1],
        [mlp_accuracy, mlp_precision, mlp_recall, mlp_f1]
    )
    
    # Plot combined confusion matrices
    print("\nPlotting combined confusion matrices...")
    plot_combined_confusion_matrices(svm_cm, mlp_cm, rf_cm)
    
    print("\nModel training, evaluation, and visualization complete!")