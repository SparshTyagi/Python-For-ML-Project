import pandas as pd
import os
import joblib
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class DataProcessor:
    @staticmethod
    def clean_and_merge_tac_data(folder_path, convert_timestamp=True):
        """Cleans and merges all TAC files from a given folder, adding a 'pid' column extracted from filenames."""
        all_data = []  # List to store dataframes

        # Iterate through all TAC files in the directory with progress
        for file in tqdm(os.listdir(folder_path), desc="Processing TAC files"):
            if file.endswith("_clean_TAC.csv"):  # Only process clean TAC files
                file_path = os.path.join(folder_path, file)
                # Extract pid from filename (everything before "_clean_TAC.csv")
                pid = file.replace("_clean_TAC.csv", "")
                try:
                    # Load the data
                    df = pd.read_csv(file_path, names=['timestamp', 'TAC_Reading'], delimiter=',', header=None)
                    # Convert timestamp to datetime
                    if convert_timestamp:
                        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

                    # Convert TAC_Reading column from string to numeric, forcing errors='coerce'
                    df['TAC_Reading'] = pd.to_numeric(df['TAC_Reading'], errors='coerce')
                    # Remove negative TAC values
                    df = df[df['TAC_Reading'] >= 0]
                    # Add 'pid' column
                    df['pid'] = pid
                    # Append cleaned dataframe to list
                    all_data.append(df)

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        # Merge all participant data into a single DataFrame
        merged_tac = pd.concat(all_data, ignore_index=True)

        return merged_tac

    @staticmethod
    def clean_accelerometer_data(file_path, pids_file, output_folder="split_accel_data", convert_timestamp=True):
        """
        Cleans accelerometer data by:
        - Splitting data into separate files per participant (pid)
        - Sorting each file by timestamp
        - Saving sorted participant-specific files to disk
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        # Load valid participant IDs
        valid_pids = pd.read_csv(pids_file, header=None, names=['pid'])
        valid_pids = valid_pids['pid'].astype(str).tolist()  # Convert to list of strings
        # Load accelerometer data
        dtype_dict = {'time': 'str', 'pid': 'str', 'x': 'str', 'y': 'str', 'z': 'str'}
        df = pd.read_csv(file_path, names=['time', 'pid', 'x', 'y', 'z'], delimiter=',', header=None,
                         dtype=dtype_dict, low_memory=False)

        # Convert timestamp to numeric
        if convert_timestamp:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            # Fix timestamps in milliseconds (if applicable)
            df.loc[df['time'] > 10 ** 10, 'time'] = df['time'] / 1000
            df['time'] = pd.to_datetime(df['time'], unit='s')

        # Convert x, y, z to numeric
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
        # Remove rows where all x, y, and z values are 0.0
        df = df[~((df['x'] == 0.0) & (df['y'] == 0.0) & (df['z'] == 0.0))]
        # Keep only valid pids
        df = df[df['pid'].astype(str).isin(valid_pids)]
        # Drop NaN values (corrupt rows)
        df = df.dropna()
        # Process each 'pid' group with progress
        for pid, pid_df in tqdm(list(df.groupby('pid')), desc="Processing accelerometer groups", total=len(valid_pids)):
            sorted_pid_df = pid_df.sort_values(by='time')
            sorted_pid_df.to_csv(f"{output_folder}/{pid}_sorted.csv", index=False)
            print(f"Saved sorted data for {pid} to {output_folder}/{pid}_sorted.csv")

        print("\nAll participant data split and sorted successfully!")

    @staticmethod
    def align_per_pid_and_save(accel_folder="split_accel_data", tac_file="cleaned_tac_data.csv",
                               output_folder="aligned_data"):
        """Aligns each participant's accelerometer data with TAC readings separately and saves them individually."""
        # Output folder
        os.makedirs(output_folder, exist_ok=True)
        # Load cleaned TAC data
        tac_df = pd.read_csv(tac_file)
        tac_df['timestamp'] = pd.to_datetime(tac_df['timestamp'], errors='coerce')
        # Process each participant with progress
        for file in tqdm(os.listdir(accel_folder), desc="Aligning accelerometer data"):
            if file.endswith("_sorted.csv"):
                pid = file.split("_")[0]  # Extract participant ID from filename
                file_path = os.path.join(accel_folder, file)
                # Load the participant's accelerometer data
                accel_df = pd.read_csv(file_path)
                accel_df['time'] = pd.to_datetime(accel_df['time'], errors='coerce')
                # Filter TAC data for the same participant
                tac_pid_df = tac_df[tac_df['pid'] == pid].copy()
                if tac_pid_df.empty:
                    print(f"No TAC data found for {pid}, skipping...")
                    continue

                # Ensure TAC data is sorted
                tac_pid_df = tac_pid_df.sort_values(by='timestamp').reset_index(drop=True)
                # Ensure accelerometer data is sorted
                accel_df = accel_df.sort_values(by='time').reset_index(drop=True)
                # Merge accelerometer data with TAC values
                merged_df = pd.merge_asof(
                    accel_df, tac_pid_df,
                    left_on='time', right_on='timestamp',  # Match accelerometer time to closest earlier TAC time
                    by='pid',
                    direction='backward'
                )

                # Assign labels: TAC ≥ 0.08 - Intoxicated (1), TAC < 0.08 - Sober (0)
                merged_df['label'] = (merged_df['TAC_Reading'] >= 0.08).astype(int)
                # Drop unnecessary columns
                merged_df = merged_df.drop(columns=['timestamp'])
                # Save aligned data for the participant
                aligned_file_path = os.path.join(output_folder, f"{pid}_aligned.csv")
                merged_df.to_csv(aligned_file_path, index=False)

                print(f"Aligned TAC values saved for {pid} at {aligned_file_path}")

        print("\nAll participant data aligned successfully!")

    @staticmethod
    def merge_aligned_data(input_folder="aligned_data", output_file="cleaned_accelerometer_data.csv"):
        """Merges all sorted participant-specific accelerometer files into a single dataset."""
        all_data = []
        # Load each sorted 'pid' file with progress
        for file in tqdm(os.listdir(input_folder), desc="Merging aligned data"):
            if file.endswith("_aligned.csv"):
                file_path = os.path.join(input_folder, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

        # Merge all participant data into a single DataFrame
        merged_df = pd.concat(all_data, ignore_index=True)
        # Ensure final dataset is sorted by participant and time
        merged_df = merged_df.sort_values(by=['pid', 'time']).reset_index(drop=True)
        # Save final dataset
        merged_df.to_csv(output_file, index=False)
        print(f"\nMerged aligned accelerometer data saved to {output_file}")

    @staticmethod
    def basic_data_explore_and_visualize(labeled_file):
        df = pd.read_csv(labeled_file)
        print(df.info())  # Check for missing values
        print(df.head())  # Preview data
        print(df.describe())  # Basic statistics
        df['label'].value_counts().plot(kind='bar')
        plt.xlabel("Class (0 = Sober, 1 = Intoxicated)")
        plt.ylabel("Count")
        plt.title("Class Distribution in Dataset")
        plt.show()


class FeatureExtractor:
    @staticmethod
    def extract_window_features(window_data):
        """Extracts statistical and spectral features from a given time window of accelerometer data."""
        if len(window_data) < 2:
            return pd.Series(dtype='float64')  # Return empty series if not enough data

        features = {}

        for axis in ['x', 'y', 'z']:
            axis_data = window_data[axis].dropna()

            if len(axis_data) < 2:
                continue  # Skip if not enough values

            features[f'{axis}_mean'] = axis_data.mean()
            features[f'{axis}_std'] = axis_data.std()
            features[f'{axis}_var'] = axis_data.var()
            features[f'{axis}_median'] = axis_data.median()
            features[f'{axis}_min'] = axis_data.min()
            features[f'{axis}_max'] = axis_data.max()
            features[f'{axis}_range'] = features[f'{axis}_max'] - features[f'{axis}_min']

            # Handle skew/kurtosis only if standard deviation is nonzero
            if features[f'{axis}_std'] > 0:
                features[f'{axis}_skew'] = np.nan_to_num(skew(axis_data, nan_policy='omit'))
                features[f'{axis}_kurtosis'] = np.nan_to_num(kurtosis(axis_data, nan_policy='omit'))
            else:
                features[f'{axis}_skew'] = 0  # Set to zero if data is constant
                features[f'{axis}_kurtosis'] = 0  # Set to zero if data is constant

            # Spectral Features (FFT)
            fft_vals = np.abs(fft(axis_data))
            features[f'{axis}_dominant_freq'] = np.argmax(fft_vals) if len(fft_vals) > 0 else 0
            features[f'{axis}_spectral_energy'] = np.sum(fft_vals ** 2) if len(fft_vals) > 0 else 0

        return pd.Series(features)

    @staticmethod
    def extract_features(df, window_size=10):
        """Extracts features using a rolling window approach."""
        feature_list = []

        # Ensure sorting before processing
        df = df.sort_values(by=['pid', 'time']).reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'])

        # Process each participant separately with progress
        for pid, pid_df in tqdm(list(df.groupby('pid')), desc="Extracting features", total=df['pid'].nunique()):
            pid_df = pid_df.set_index('time')

            # Iterate through rolling windows manually
            start_time = pid_df.index.min()
            end_time = pid_df.index.max()
            current_time = start_time

            while current_time <= end_time:
                # Define the window range
                window_start = current_time
                window_end = current_time + pd.Timedelta(seconds=window_size)
                window_data = pid_df[(pid_df.index >= window_start) & (pid_df.index < window_end)]

                if len(window_data) < 2:
                    current_time += pd.Timedelta(seconds=window_size)  # Move to next window
                    continue  # Skip if not enough data points

                # Extract features from this window
                feature_dict = FeatureExtractor.extract_window_features(window_data)
                feature_dict['pid'] = pid
                feature_dict['time'] = window_start

                # Assign the majority label in the window
                feature_dict['label'] = window_data['label'].mode()[0] if not window_data['label'].mode().empty else 0

                feature_list.append(feature_dict)

                # Move to the next window
                current_time += pd.Timedelta(seconds=window_size)

        # Convert list to DataFrame
        feature_df = pd.DataFrame(feature_list)

        return feature_df


class RandomForestModel:
    @staticmethod
    def train_random_forest(X, y):
        """Trains and evaluates a Random Forest classifier."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        # Calculate total iterations:
        # total iterations = n_estimators * max_depth * min_samples_split * number_of_cv_folds
        total_iter = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * 5

        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)

        # Set up Grid Search with cross-validation
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)

        # Wrap grid_search.fit call with tqdm_joblib to track progress
        with tqdm_joblib(tqdm(total=total_iter, desc="GridSearchCV iterations")):
            grid_search.fit(X_train, y_train)

        # Get the best model
        best_rf = grid_search.best_estimator_

        # Make predictions
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)

        # Print evaluation results
        print("\nBest Parameters:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")

        # Compute feature importance
        feature_importance = (pd.DataFrame({'Feature': X.columns, 'Importance': best_rf.feature_importances_})
                              .sort_values(by='Importance', ascending=False))

        return best_rf, y_test, y_pred, feature_importance, accuracy

    @staticmethod
    def train_random_forest_new_hyperparameters(X, y):
        """Trains and evaluates a Random Forest classifier with new hyperparameters."""
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [20, 30, 40],
            'min_samples_split': [1, 2, 5],
            'class_weight': ['balanced']
        }

        # Calculate total iterations:
        # total iterations = n_estimators * max_depth * min_samples_split * number_of_cv_folds
        total_iter = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * 5

        # Initialize Random Forest Classifier
        rf = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation and tqdm_joblib progress
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        with tqdm_joblib(tqdm(total=total_iter, desc="GridSearchCV iterations (new hyperparameters)")):
            grid_search.fit(X_train, y_train)

        # Get the best model
        best_rf = grid_search.best_estimator_

        # Make predictions
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sober_accuracy = tn / (tn + fp)

        # Print evaluation results
        print("\nBest RF Parameters:", grid_search.best_params_)
        print("Accuracy:", accuracy)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Sober Accuracy: {sober_accuracy:.4f}")

        # Compute feature importance
        feature_importance = (pd.DataFrame({'Feature': X.columns, 'Importance': best_rf.feature_importances_})
                                .sort_values(by='Importance', ascending=False))

        return best_rf, y_test, y_pred, feature_importance, accuracy

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, accuracy, save_path="results/confusion_matrix_new_hyperparameters.png"):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Sober', 'Intoxicated'], yticklabels=['Sober', 'Intoxicated'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"RandomForest Confusion Matrix (Accuracy: {accuracy:.4f})")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_importance, save_path="results/feature_importance_new_hyperparameters.png"):
        """Plots and saves the top 10 most important features."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance[:10])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature Name")
        plt.title("Top 10 Important Features in Random Forest")

        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    # Uncomment the following steps as needed.

    # TAC data preprocessing
    # folder_path = "data/clean_tac"
    # cleaned_tac = DataProcessor.clean_and_merge_tac_data(folder_path)
    # cleaned_tac.to_csv("cleaned_tac_data.csv", index=False)
    #
    # Accelerometer data preprocessing
    # file_path = "data/all_accelerometer_data_pids_13.csv"
    # pids_file = "data/pids.txt"
    # DataProcessor.clean_accelerometer_data(file_path, pids_file, output_folder="split_accel_data")
    # DataProcessor.align_per_pid_and_save(accel_folder="split_accel_data", tac_file="cleaned_tac_data.csv", output_folder="aligned_data")
    # DataProcessor.merge_aligned_data(input_folder="aligned_data", output_file="final_labeled_data.csv")
    #
    # df = pd.read_csv("final_labeled_data.csv")
    # feature_df = FeatureExtractor.extract_features(df, window_size=10)
    # feature_df.to_csv("extracted_features.csv", index=False)
    #
    df = pd.read_csv("extracted_features.csv")
    # Drop non-relevant columns
    X = df.drop(columns=['pid', 'time', 'label'], errors='ignore')
    y = df['label']

    # Train Random Forest Model (with GridSearchCV progress shown)
    best_rf, y_test, y_pred, feature_importance, accuracy = RandomForestModel.train_random_forest(X, y)
    
    # Save the trained Random Forest model into a folder (e.g., Best_Models)
    model_save_dir = "Best_Models"
    os.makedirs(model_save_dir, exist_ok=True)
    rf_model_path = os.path.join(model_save_dir, "best_rf_model.pkl")
    joblib.dump(best_rf, rf_model_path)
    print(f"Best RF model saved to {rf_model_path}")
    
    # Save the confusion matrix image
    confusion_matrix_save_path = "results/rf_confusion_matrix.png"
    RandomForestModel.plot_confusion_matrix(y_test, y_pred, accuracy, save_path=confusion_matrix_save_path)
    print(f"Confusion matrix saved to {confusion_matrix_save_path}")
    
    # Save the feature importance plot
    feature_importance_save_path = "results/rf_feature_importance.png"
    RandomForestModel.plot_feature_importance(feature_importance, save_path=feature_importance_save_path)
    print(f"Feature importance plot saved to {feature_importance_save_path}")    

    # You can also train SVM and MLP as needed:
    # SVM.train_svm(X, y)
    # MLP.train_mlp(X, y)

    # Plot evaluation results:
    # RandomForestModel.plot_confusion_matrix(y_test, y_pred, accuracy)
    # RandomForestModel.plot_feature_importance(feature_importance)