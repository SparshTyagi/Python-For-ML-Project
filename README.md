# Python For ML Project

This project explores various machine learning techniques for activity classification and regression using sensor data. It includes data processing, feature extraction, model training, hyperparameter tuning, and model comparison. Machine learning models implemented in this project include:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **LightGBM**
- **CatBoost**
- **Random Forests**
- **XGBoost**
- **Multi-Layer Perceptron (MLP)**

## Project Structure

- **Data Processing and Feature Extraction**
  - `RandomForestsActivityClassification.py`: Contains functions to clean and merge TAC and accelerometer data, align TAC readings with accelerometer signals, extract features (statistical and spectral) using a rolling window approach, and train Random Forest models.
  - Other utility functions for data cleaning and exploration are embedded within this file.

- **Model Training**
  - `LogisticRegression.py`: Implements Logistic Regression with hyperparameter tuning and feature importance visualization.
  - `KNN.py`: Implements KNN with randomized hyperparameter search and progress visualization.
  - `LightGBM.py`: Implements LightGBM classifier with hyperparameter search.
  - `CatBoost.py`: Implements CatBoost classifier with early stopping in randomized search.
  - `XGB-MLP.py`: Contains the skeleton for XGBoost and MLP classifiers and functions to compare model performance.
  
- **Model Comparison**
  - `ModelComparisons.ipynb`: A Jupyter Notebook that loads saved best models from the `Best_Models` folder, evaluates them on a common test set, and displays a comparison table of key metrics (Accuracy, Precision, Recall, F1 Score).

- **Saved Models and Results**
  - **Best_Models/** folder: Contains the saved pickle files for each best-performing model.
  - **results/** folder: Contains plots for confusion matrices and feature importance visualizations.

## Requirements

- Python 3.12 (or greater)
- Required packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `seaborn`
  - `tqdm` and `tqdm_joblib`
  - `lightgbm`
  - `catboost`
  - `xgboost` (if using XGBoost portions)
- Install the dependencies with:
  ```
  pip install -r requirements.txt
  ```

## Setup and Usage

1. **Data Preprocessing**
   - Run the data cleaning, alignment, and feature extraction functions in `RandomForestsActivityClassification.py` to generate your dataset (`extracted_features.csv`).  
   - For example, uncomment and run the preprocessing sections at the bottom of the file.

2. **Model Training**
   - Train models individually by running the respective scripts. For example:
     - Logistic Regression: `python LogisticRegression.py`
     - KNN: `python KNN.py`
     - LightGBM: `python LightGBM.py`
     - CatBoost: `python CatBoost.py`
     - Random Forest: `python RandomForestsActivityClassification.py`
   - Each script includes hyperparameter tuning and automatically saves:
     - The best model (saved as a pickle file in the `Best_Models` folder)
     - Confusion Matrix plots
     - Feature importance visualizations

3. **Model Comparisons**
   - Open and run `ModelComparisons.ipynb` in Jupyter Notebook to load all models from `Best_Models`, evaluate them on a common test set, and compare their performance metrics.

## Additional Notes

- The project uses `tqdm` and `tqdm_joblib` for progress visualization during grid searches and feature extraction.
- Early stopping is implemented in the CatBoost training process to reduce training time.
- Environment variable `LOKY_MAX_CPU_COUNT` can be adjusted to limit the number of cores used by joblib. This is set in `ModelComparisons.ipynb`.

## Acknowledgments

This project integrates techniques and models from popular machine learning libraries such as scikit-learn, LightGBM, CatBoost, and XGBoost. Contributions and suggestions are welcome!

Happy modeling!