# Coursera_Project__Scikit-Learn_to_Solve_Regression_Machine_Learning_Problems

## Project Overview
This project applies regression machine learning techniques to predict a student's chance of university admission. An XGBoost regression model is trained using academic and application-related features, then evaluated and refined through hyperparameter tuning and feature selection.

The workflow includes:

- Loading and inspecting the university admission dataset.
- Exploring correlations between numerical features.
- Splitting the data into training and testing sets.
- Tuning an XGBoost regression model with randomized hyperparameter search and cross-validation.
- Evaluating model performance using R², MSE, RMSE, and MAE.
- Visualizing predicted versus actual admission chances.
- Identifying feature importance and selecting the strongest feature subset using cross-validation.

## Data Understanding
The project uses the `university_admission.csv` dataset, which contains applicant characteristics and their corresponding chance of admission.

Key features include:

1. **GRE Score**: Graduate Record Examination score.
2. **TOEFL Score**: Test of English as a Foreign Language score.
3. **University Rating**: Rating of the university attended.
4. **SOP**: Strength of Statement of Purpose.
5. **LOR**: Strength of Letter of Recommendation.
6. **CGPA**: Undergraduate cumulative grade point average.
7. **Research**: Whether the applicant has research experience.

The target variable is:

- **Chance_of_Admission**: The predicted likelihood of admission to a university program.

## Modeling and Evaluation
An `XGBRegressor` model from the XGBoost library was used to predict admission chances. Hyperparameters were optimized with `RandomizedSearchCV` using five-fold cross-validation and RMSE as the tuning metric.

Model performance was evaluated on the test set using:

- **R² Score**: Measures the proportion of variation in admission chance explained by the model.
- **Mean Squared Error (MSE)**: Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE)**: Expresses prediction error in the target variable's units.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.

Feature importance analysis showed that all available variables contributed to prediction performance. GRE Score and CGPA were identified as the most influential predictors of admission chance.

## Results
The final XGBoost model achieved a test R² score of **0.9266**, improving on the earlier model score of **0.9105**. This means the model explains approximately **92.66%** of the variation in university admission chances.

A small difference between training and testing scores indicated slight overfitting, but the final model still generalized well to unseen test data.

## Conclusion
The project demonstrates that XGBoost regression can accurately predict university admission chances from academic performance and application characteristics. GRE Score and CGPA were the strongest predictors, while the remaining variables also provided useful information.

The final model provides strong predictive performance and can be used as a practical example of regression modeling, hyperparameter optimization, cross-validation, and feature selection with Scikit-learn and XGBoost.