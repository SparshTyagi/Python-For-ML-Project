import pandas as pd
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb  # New import for XGBoost


# Replacing SVM with XGB
class XGB:
    @staticmethod
    def train_xgb(X, y):
        """Trains an XGBoost classifier with iterative checkpointing during hyperparameter search."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        checkpoint_path = "best_xgb_model.pkl"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint XGB model for further improvement...")
            best_xgb = joblib.load(checkpoint_path)
            best_score = np.mean(cross_val_score(best_xgb, X_train, y_train,
                                                 cv=cv, scoring='accuracy', n_jobs=-1))
            print(f"Current checkpoint model CV score: {best_score:.4f}")
        else:
            best_score = -np.inf
            best_xgb = None

        param_distributions = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        grid = list(ParameterGrid(param_distributions))
        total_iter = len(grid) * 5
        
        for i, params in enumerate(tqdm(grid, desc="XGB Hyperparameter Search", ncols=80), 1):
            candidate = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
            score = np.mean(cross_val_score(candidate, X_train, y_train,
                                              cv=cv, scoring='accuracy', n_jobs=-1))
            if score > best_score:
                best_score = score
                best_xgb = candidate.fit(X_train, y_train)
                joblib.dump(best_xgb, checkpoint_path)
                print(f"XGB checkpoint: iteration {i} / {len(grid)}, best score: {best_score:.4f}")
            if i % 10 == 0:
                joblib.dump(best_xgb, checkpoint_path)
                print(f"XGB checkpoint: iteration {i} / {len(grid)}, best score: {best_score:.4f}")
        joblib.dump(best_xgb, checkpoint_path)
        print(f"Best XGB model saved to {checkpoint_path}")
        
        y_pred = best_xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        print("\nBest XGB Model:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")
        return best_xgb, y_test, y_pred, accuracy, precision, recall, f1

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/xgb_confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"XGB Confusion Matrix (Accuracy: {accuracy:.4f})")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        
    @staticmethod
    def plot_feature_importance(model, X, save_path="results/xgb_feature_importance.png"):
        """Plots and saves XGB feature importance."""
        importance = model.feature_importances_
        features = X.columns
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (XGB)')
        plt.xlabel('Importance')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


class MLP:
    @staticmethod
    def train_mlp(X, y):
        """Trains an MLP classifier with iterative checkpointing during hyperparameter search."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        checkpoint_path = "best_mlp_model.pkl"
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        grid = list(ParameterGrid(param_grid))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        best_score = -np.inf
        best_mlp = None
        
        for i, params in enumerate(tqdm(grid, desc="MLP Hyperparameter Search", ncols=80), 1):
            candidate = MLPClassifier(max_iter=500, random_state=42, **params)
            score = np.mean(cross_val_score(candidate, X_train, y_train,
                                              cv=cv, scoring='accuracy', n_jobs=-1))
            if score > best_score:
                best_score = score
                best_mlp = candidate.fit(X_train, y_train)
                joblib.dump(best_mlp, checkpoint_path)
                print(f"MLP checkpoint: iteration {i} / {len(grid)}, best score: {best_score:.4f}")
            if i % 10 == 0:
                joblib.dump(best_mlp, checkpoint_path)
                print(f"MLP checkpoint: iteration {i} / {len(grid)}, best score: {best_score:.4f}")
        joblib.dump(best_mlp, checkpoint_path)
        print(f"Best MLP model saved to {checkpoint_path}")
        
        y_pred = best_mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)
        
        print("\nBest MLP Parameters:", best_mlp)
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
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
        plt.title('Top 10 Features (MLP)')
        plt.xlabel('Permutation Importance')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


def compare_model_performance(rf_metrics, xgb_metrics, mlp_metrics, save_path="results/model_comparison.png"):
    """Compares and visualizes the performance metrics of different models."""
    models = ['Random Forest', 'XGBoost', 'MLP']
    metrics = {
        'Accuracy': [rf_metrics[0], xgb_metrics[0], mlp_metrics[0]],
        'Precision': [rf_metrics[1], xgb_metrics[1], mlp_metrics[1]],
        'Recall': [rf_metrics[2], xgb_metrics[2], mlp_metrics[2]],
        'F1 Score': [rf_metrics[3], xgb_metrics[3], mlp_metrics[3]]
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


def plot_combined_confusion_matrices(xgb_cm, mlp_cm, rf_cm=None, save_path="results/combined_confusion_matrices.png"):
    """Plots and saves combined confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3 if rf_cm is not None else 2, figsize=(15, 5))
    # Plot XGBoost confusion matrix
    sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
    axes[0].set_title("XGBoost")
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
    X = df.drop(columns=['pid', 'time', 'label'], errors='ignore')
    y = df['label']
    
    print("\n===== XGBoost Model Training =====")
    best_xgb, xgb_y_test, xgb_y_pred, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1 = XGB.train_xgb(X, y)
    xgb_cm = confusion_matrix(xgb_y_test, xgb_y_pred)
    XGB.plot_confusion_matrix(xgb_y_test, xgb_y_pred, xgb_accuracy)
    XGB.plot_feature_importance(best_xgb, X)
    
    xgb_model_path = "best_xgb_model.pkl"
    joblib.dump(best_xgb, xgb_model_path)
    print(f"Best XGBoost model saved to {xgb_model_path}")
    
    print("\n===== MLP Model Training =====")
    best_mlp, mlp_y_test, mlp_y_pred, mlp_accuracy, mlp_precision, mlp_recall, mlp_f1 = MLP.train_mlp(X, y)
    mlp_cm = confusion_matrix(mlp_y_test, mlp_y_pred)
    MLP.plot_confusion_matrix(mlp_y_test, mlp_y_pred, mlp_accuracy)
    print("\nCalculating MLP feature importance...")
    MLP.plot_feature_importance(best_mlp, X, y)
    
    mlp_model_path = "best_mlp_model.pkl"
    joblib.dump(best_mlp, mlp_model_path)
    print(f"Best MLP model saved to {mlp_model_path}")
    
    print("\n===== Comparing Model Performance =====")
    # For demonstration purposes, use placeholder RF metrics
    rf_accuracy = 0.835
    rf_precision = 0.691
    rf_recall = 0.585
    rf_f1 = 0.634
    rf_cm = np.array([[5386, 494], [785, 1108]])
    
    compare_model_performance(
        [rf_accuracy, rf_precision, rf_recall, rf_f1],
        [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1],
        [mlp_accuracy, mlp_precision, mlp_recall, mlp_f1]
    )
    
    print("\nPlotting combined confusion matrices...")
    plot_combined_confusion_matrices(xgb_cm, mlp_cm, rf_cm)
    
    print("\nModel training, evaluation, and visualization complete!")
    
    
    
# Best XGB Model:
# Accuracy: 0.8337
# Precision: 0.6775
# Recall: 0.6049
# F1 Score: 0.6391
# Sober Accuracy: 0.9073