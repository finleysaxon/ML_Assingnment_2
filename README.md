# Airline Passenger Satisfaction — ML Classification

## a. Problem Statement

The goal of this project is to predict **airline passenger satisfaction** (satisfied vs. neutral or dissatisfied) based on various travel and service-quality features. This is a **binary classification** problem. By building and comparing multiple machine learning models, we aim to identify the most effective classifier and understand which factors contribute most strongly to passenger satisfaction.

## b. Dataset Description

- **Source**: [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- **Type**: Binary Classification
- **Target Variable**: `satisfaction` (satisfied / neutral or dissatisfied)
- **Total Instances**: 129,880 (train: 103,904 | test: 25,976)
- **Total Features**: 22 (excluding target and ID columns)
- **Feature Types**: Numerical (Age, Flight Distance, Delays) + Ordinal service ratings (1–5) + Categorical (Gender, Customer Type, Travel Type, Class)

| Feature Category | Examples | Count |
|-----------------|----------|-------|
| Demographics | Age, Gender, Customer Type | 3 |
| Travel Info | Type of Travel, Class, Flight Distance | 3 |
| Service Ratings (1-5) | Inflight wifi, Seat comfort, Online boarding, Entertainment, etc. | 14 |
| Delay Metrics | Departure Delay in Minutes, Arrival Delay in Minutes | 2 |

This dataset satisfies the assignment requirements of **minimum 12 features** and **minimum 500 instances**.

## c. Models Used

All 6 classification models were implemented on the same dataset. Evaluation metrics were calculated on the **test set** (25,976 instances).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8781 | 0.9296 | 0.8730 | 0.8406 | 0.8565 | 0.7510 |
| Decision Tree | 0.9449 | 0.9439 | 0.9366 | 0.9361 | 0.9363 | 0.8878 |
| kNN | 0.9272 | 0.9685 | 0.9501 | 0.8779 | 0.9125 | 0.8522 |
| Naive Bayes | 0.8625 | 0.9214 | 0.8584 | 0.8174 | 0.8374 | 0.7191 |
| Random Forest (Ensemble) | 0.9617 | 0.9936 | 0.9730 | 0.9376 | 0.9550 | 0.9222 |
| XGBoost (Ensemble) | **0.9619** | **0.9948** | **0.9704** | **0.9407** | **0.9553** | **0.9224** |

**Best Model**: XGBoost (highest F1 Score = 0.9553)

### Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression acts as a reliable benchmark model, achieving an accuracy of 87.81%. Since it is a linear classifier, it has limitations in modeling complex, non-linear relationships among service quality attributes and passenger satisfaction. This limitation is reflected in its comparatively lower MCC score (0.7510). However, with an AUC of 0.9296, it still demonstrates decent ability to distinguish between satisfied and dissatisfied passengers. |
| Decision Tree | The Decision Tree model performs strongly with an accuracy of 94.49%. Its precision and recall values are closely aligned (0.9366 and 0.9361), indicating balanced classification behavior. Although the AUC score (0.9439) is slightly lower compared to some other models, it still shows good discriminatory power. One concern with this model is its tendency to overfit, especially when not carefully pruned. |
| kNN | The kNN model achieves high precision (0.9501), meaning it makes fewer false-positive predictions. However, its recall (0.8779) is comparatively lower, suggesting it may miss some truly satisfied passengers. The strong AUC value (0.9685) indicates that the model ranks predictions effectively. Its performance depends heavily on proper feature scaling, which was handled using StandardScaler. |
| Naive Bayes | Naive Bayes shows the weakest overall performance, with an accuracy of 86.25% and an F1-score of 0.8374. This is likely due to its assumption of feature independence, which does not hold well in this dataset since many service-related features are correlated. Even so, the model achieves an AUC of 0.9214, indicating that it still captures general classification patterns reasonably well. |
| Random Forest (Ensemble) | Random Forest delivers excellent overall results, reaching 96.17% accuracy and an impressive AUC of 0.9936. By aggregating predictions from 100 decision trees, it reduces overfitting and improves generalization. The high precision score (0.9730) highlights its reliability in predicting satisfied passengers. Its performance is very close to XGBoost, although slightly lower in F1-score and MCC. |
| XGBoost (Ensemble) | XGBoost emerges as the top-performing model across nearly all evaluation metrics. With 96.19% accuracy, an AUC of 0.9948, F1-score of 0.9553, and MCC of 0.9224, it consistently outperforms the other models. The boosting mechanism allows it to model complex feature interactions effectively. The improvement over Random Forest is modest but consistent, reinforcing the advantage of gradient boosting on structured datasets like this one. |

## Project Structure

```
ML_Assingnment_2/
│── app.py                          # Streamlit app entry point
│── requirements.txt                # Python dependencies
│── README.md                       # Project documentation
│── model/                          # Model files & notebooks
│   ├── resources/               # Trained model pickle files
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── knn.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   ├── scaler.pkl
│   │   └── feature_columns.pkl
│   ├── model_training.ipynb        # Model training notebook
│   ├── data_preprocessing.ipynb    # Data preprocessing notebook
│   ├── Eda.ipynb                   # Exploratory Data Analysis notebook
│   ├── metric_calculation.py       # Evaluation metric utilities
│   └── train_split.py              # Train/test split utilities
│── dataset/
│   ├── train.csv
│   ├── test_original.csv
│   └── test_sample.csv
│── reports/                        # JSON reports & static visualizations
```

## Streamlit App Features

- **CSV Upload**: Upload test data for on-the-fly evaluation
- **Model Selection Dropdown**: Choose from 6 trained classifiers
- **Evaluation Metrics Dashboard**: Accuracy, AUC, Precision, Recall, F1, MCC displayed per model
- **Confusion Matrix**: Interactive heatmap for each selected model
- **Visual Comparison**: Side-by-side metric comparison across all models
- **EDA Insights**: Pre-computed exploratory analysis visualizations

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live App

> [Streamlit App Link — to be added after deployment]

## Technologies Used

- Python 3.12
- Streamlit
- scikit-learn
- XGBoost
- pandas, NumPy
- matplotlib, seaborn
