# Classification for Marketing Campaigns Subscription

This repository contains machne learning classifier algos toidentify customers likely to subscribe to a marketing campaign using historical data on demographics, behavior, and campaign engagement.

## Python Notebook
You can access the Jupyter notebook [here](https://github.com/AICarope/Classification-for-Marketing-Campaings-Subscription/blob/main/1.EDA%26ML.ipynb)

## Business Understanding

### Objective: 
Compare the results of k-nearest neighbors, logistic regression, decision trees, and support vector machines

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
4. **Identify X (independent) and Y (dependent, target)**:
   X numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
     categorcial converted to numerical= ['job_blue-collar', 'marital_married', 'education_secondary', 'housing_yes', 'loan_yes']
   Y categorcial but converted to integer for analysis: clients who subscribed (1) Yes and those who didn’t (0) No.
   
   
6. **Categorical Encoding**: Converted categorical features into numeric using one-hot encoding.
Shape of X before dropping: (4521, 42)
Shape of X after dropping: (4521, 35)
Training set shape: (3616, 35), Testing set shape: (905, 35)
 

### Modeling
- Applied multiple classification algorithms to predict subscription likelihood.
- Models used (Classifier Modifications):
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)

## Feature Importance

### Top Features Identified (Feauture Selection)
Use insights from the visualizations, correlation matrix, and feature importance analysis to select the most relevant features.
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

### Model Performance (Model Traning)

| Model                   | Accuracy (%) |
|-------------------------|--------------|
| Logistic Regression     | 89.94        |
| Support Vector Machine  | 89.06        |
| K-Nearest Neighbors     | 87.73        |
| Decision Tree           | 86.52        |

### Best-Performing Model
- **Logistic Regression** achieved the highest accuracy (89.94%).
- **Insights**:
  - **KNN** (87.73%) and SVM (89.06%) performed well, suggesting potential non-linear relationships in the data.
  - **Decision Tree** (86.52%) underperformed slightly, likely due to overfitting or suboptimal splitting criteria.

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
- **Subscription**:
  - `Yes` 11.52% (521) people subscribed.
  - `No` 88.4% (4000) did not subscribed.
    
![image](https://github.com/user-attachments/assets/28119ccf-ec39-4ea1-845d-4a736725ad0e)

- **Critical Predictors**:
  Features below have a strong influence on the Y variable. DUration may be the most important feauture.
  - `duration`: Longer call durations correlate strongly with subscriptions.
  - `poutcome_success`: Past successful interactions increase likelihood.
    
- **Recommendation**:
  - Focus on long conversations with customers as they are more likely to lead to subscriptions.
  - Target customers with a history of success in previous campaigns (poutcome_success).
  - October and March show higher correlation with subscriptions. The bank could increase efforts during these months.
  - Customers with tertiary education and single individuals show a higher likelihood of subscribing. These groups should be prioritized in campaigns.
    
- **Targeted Marketing**:
  - Focus on clients with longer call durations and a history of successful outcomes.

### Future Deployment Steps
1. Build a deployment pipeline for real-time prediction.
2. Integrate the model into marketing systems.
3. Continuously monitor and update the model to maintain performance.
4. Investigate interactions between features (e.g., combining features like poutcome_success and duration).
5. Combining multiple models using techniques like Random Forest, Gradient Boosting (e.g., XGBoost).
6. Potential improvements, such as hyperparameter tuning, additional feature engineering, or trying ensemble methods.
7. SVM with the rbf kernel is computationally expensive, especially on large datasets. When running this data set with SVM it took more than one hour and results were completed.
   
## Technologies Used

### Programming Languages
- Python

### Libraries
- Pandas, NumPy, Matplotlib, Scikit-learn, Seaborn

### Tools
- Version Control: Git/GitHub

---
For detailed code and analysis, refer to the repository files. Feedback and contributions are welcome!
