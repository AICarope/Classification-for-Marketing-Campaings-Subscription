# Classification for Marketing Campaigns Subscription

This repository contains machne learning classifier algos toidentify customers likely to subscribe to a marketing campaign using historical data on demographics, behavior, and campaign engagement.

## Python Notebook
You can access the Jupyter notebook [here](https://github.com/AICarope/Classification-for-Marketing-Campaings-Subscription/blob/main/1.EDA%26ML.ipynb)

## Business Understanding

### Objective**: Compare the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines

- **Goal**: Classify customers likely to subscribe to a campaign based on historical data.
            To compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines)
- **Data Understanding**: Analyze customer demographics, past behavior, and engagement with previous campaigns.
- **Initial Analysis**: Conduct exploratory data analysis (EDA) to identify trends and patterns in the data.

## Data Preparation
-The data set contains **4521 rows** and **43 columns** 
### Steps Taken
1. **Data Cleaning**: Inspection of duplicates and handled missing values.
2. **Feature Engineering**: Created additional features based on domain knowledge.
3. **Scaling**: Standardized numerical features for model compatibility.
4. **Categorical Encoding**: Converted categorical features into numeric using one-hot encoding.
Shape of X before dropping: (4521, 42)
Shape of X after dropping: (4521, 35)
Training set shape: (3616, 35), Testing set shape: (905, 35)

### Modeling
- Applied multiple classification algorithms to predict subscription likelihood.
- Models used:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)

## Feature Importance

### Top Features Identified
The following features were found to have the highest correlation with the target variable (`y`):

| Feature             | Correlation |
|---------------------|-------------|
| duration            | 0.401       |
| poutcome_success    | 0.283       |
| month_oct           | 0.146       |
| previous            | 0.117       |
| pdays               | 0.104       |

![Feature Correlation Heatmap](https://github.com/user-attachments/assets/e187c0b0-173b-443b-8b5f-71fd737d5313)

## Evaluation

### Model Performance

| Model                   | Accuracy (%) |
|-------------------------|--------------|
| Logistic Regression     | 89.94        |
| Support Vector Machine  | 89.06        |
| K-Nearest Neighbors     | 87.73        |
| Decision Tree           | 86.52        |

### Best-Performing Model
- **Logistic Regression** achieved the highest accuracy (89.94%).
- **Insights**:
  - KNN (87.73%) and SVM (89.06%) performed well, suggesting potential non-linear relationships in the data.
  - Decision Tree (86.52%) underperformed slightly, likely due to overfitting or suboptimal splitting criteria.

### Hyperparameter Tuning

| Model                   | Best Parameters                                                     | Accuracy (%) |
|-------------------------|---------------------------------------------------------------------|--------------|
| Logistic Regression     | {'C': 1, 'solver': 'lbfgs'}                                        | 89.94        |
| Decision Tree           | {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 4}    | 89.38        |
| K-Nearest Neighbors     | {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'}    | 88.16        |

### Metrics Used
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Key Findings**:
  - Logistic Regression offers a balance between accuracy and interpretability.
  - Decision Trees provide marginal accuracy trade-offs but capture non-linear relationships.

## Deployment and Recommendations

### Model Recommendation
- **Preferred Model**: Logistic Regression.
  - Highest accuracy (89.94%).
  - Simple and interpretable for stakeholders.
  - Efficient for real-time predictions.
- **Alternative**: Decision Tree for scenarios requiring non-linear insights, with a marginal accuracy trade-off (89.38%).

### Business Insights
- **Critical Predictors**:
  - `duration`: Longer call durations correlate strongly with subscriptions.
  - `poutcome_success`: Past successful interactions increase likelihood.
- **Targeted Marketing**:
  - Focus on clients with longer call durations and a history of successful outcomes.

### Deployment Steps
1. Build a deployment pipeline for real-time prediction.
2. Integrate the model into marketing systems.
3. Continuously monitor and update the model to maintain performance.

## Technologies Used

### Programming Languages
- Python

### Libraries
- Pandas, NumPy, Matplotlib, Scikit-learn

### Tools
- Version Control: Git/GitHub

---
For detailed code and analysis, refer to the repository files. Feedback and contributions are welcome!
