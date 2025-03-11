# Python For ML Project

This project is the final submission for the Python for Machine Learning and Visualization course at the Technical University of Munich, Germany. It aims to classify heavy drinking behavior using accelerometer data from smartphones through the application of Random Forest algorithms and other machine learning techniques.

## Project Overview

The goal of this project is to reproduce and potentially improve on the results presented in the associated conference paper by developing our own methodology using a cleanroom approach. We focus on classifying heavy drinking behavior based on sensor data.

## Team Members

- Sparsh Tyagi  
- Rada  
- Qinchuan  

## Dataset

We use the **Bar Crawl: Detecting Heavy Drinking** dataset from the UCI Machine Learning Repository. This dataset includes:
- Accelerometer data from smartphone sensors at a 40 Hz sampling rate.
- Transdermal Alcohol Content (TAC) measurements from ankle bracelets.
- Data collected from 13 undergraduate students during a drinking event.

For more details, please refer to [UCI Repository](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking).

## Objective

Our primary goal is to reproduce and potentially improve upon the published results in the corresponding conference paper. We compare our findings with the published results while developing our own methodology using a cleanroom approach.

## Methodology

### Data Preprocessing
- Cleaning and merging TAC and accelerometer data.
- Aligning TAC readings with accelerometer signals.
- Feature extraction using a rolling window approach.

### Feature Engineering
- Statistical features (mean, standard deviation, variance, etc.)
- Spectral features (dominant frequency, spectral energy, etc.)

### Model Implementation
- **Primary Model:** Random Forests
- Additional models for comparison:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - LightGBM
  - CatBoost
  - XGBoost
  - Multi-Layer Perceptron (MLP)

### Model Evaluation
- Cross-validation and hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
- Performance metrics: Accuracy, Precision, Recall, F1 Score.

## Project Structure

- **Data Processing and Feature Extraction**
  - `RandomForestsActivityClassification.py`: Contains functions to clean and merge TAC and accelerometer data, align TAC readings with accelerometer signals, extract features (statistical and spectral) using a rolling window approach, and train Random Forest models. Other utility functions for data cleaning and exploration are also embedded in this file.
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
     - Confusion matrix plots
     - Feature importance visualizations

3. **Model Comparisons**
   - Open and run `ModelComparisons.ipynb` in Jupyter Notebook to load all models from `Best_Models`, evaluate them on a common test set, and compare their performance metrics.

## Additional Notes

- The project uses `tqdm` and `tqdm_joblib` for progress visualization during grid searches and feature extraction.
- Early stopping is implemented in the CatBoost training process to reduce training time.
- Environment variable `LOKY_MAX_CPU_COUNT` can be adjusted to limit the number of cores used by joblib. This is set in `ModelComparisons.ipynb`.

## Results

Below is an example comparison of the best models obtained in this project:

```
Comparison of Best Models:
             Model File  Accuracy  Precision   Recall  F1 Score
    best_lgbm_model.pkl  0.838415   0.685931 0.620708  0.651692
best_catboost_model.pkl  0.835585   0.687158 0.596408  0.638575
      best_rf_model.pkl  0.834427   0.683636 0.595880  0.636749
     best_xgb_model.pkl  0.833655   0.677515 0.604860  0.639129
     best_mlp_model.pkl  0.760710   0.522696 0.200740  0.290076
      best_lr_model.pkl  0.752348   0.354545 0.020602  0.038942
```

## Future Work

- Explore deep learning approaches for time series classification.
- Investigate the impact of different feature engineering techniques.
- Analyze temporal patterns of drinking behavior to further improve the classification.

## Acknowledgments

This project integrates techniques and models from popular machine learning libraries such as scikit-learn, LightGBM, CatBoost, and XGBoost. We thank the original authors of the Bar Crawl dataset and the associated conference paper for providing the foundation for this study.

## References

1. [Bar Crawl: Detecting Heavy Drinking Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking)  
2. [Conference Paper Reference](https://ceur-ws.org/Vol-2429/paper6.pdf)

Happy modeling!