# Advanced-Regression-Methods
Build a robust and generalizable model capable of predicting y for unseen data (Xtest) by exploring and comparing several modern regression approaches. Feature engineering and model regularization / Comparative analysis of regression techniques / RMSE-based model evaluation and optimization

# Advanced Regression Methods

## Project Overview

This project was part of the **Advanced Regression Methods** course, aiming to **build the most accurate predictive model** for a target variable `y` based on 100 explanatory variables `x(1), …, x(100)`.

The main challenge was to **minimize prediction error** on a test dataset (`Xtest.txt`), using the **Root Mean Square Error (RMSE)** as the evaluation metric.

---

## Objective

Develop a robust and efficient regression model capable of predicting the variable `y` from 100 explanatory features.  
The final model should generate **100 predictions** corresponding to the observations in the provided test dataset.

---

## Project Steps

1. **Data Loading and Exploration**
   - Descriptive analysis of the target `y` and predictors `x(i)`
   - Computation of a baseline RMSE using the mean of `y`

2. **Model Selection and Testing**
   - Multiple Linear Regression (MLR)
   - **Ridge Regression**
   - **LASSO**
   - **Principal Component Regression (PCR)**
   - **Partial Least Squares (PLS)**
   - **Stepwise Regression**

3. **Model Evaluation**
   - Cross-validation for performance comparison
   - Selection of the optimal model based on minimum RMSE

4. **Final Predictions**
   - Application of the selected model to `Xtest.txt`
   - Export of the resulting predictions in `NAME.txt`

---

## Tools & Technologies

| Tool | Purpose |
|------|----------|
| **Python / R** | Data analysis and modeling |
| **NumPy / pandas** | Data manipulation and preprocessing |
| **scikit-learn** | Implementation of regression models |
| **matplotlib / seaborn** | Visualization of results |
| **statsmodels** | Statistical modeling and diagnostics |

---

## Evaluation Metric

The performance metric used is the **Root Mean Square Error (RMSE)**:

RMSE = sqrt( (1/n_test) * Σ_{i=1}^{n_test} (y_i - ŷ_i)^2 )

A lower RMSE indicates a more accurate and stable predictive model.

---

## Skills Developed

- Mastery of **advanced regression techniques**
- Handling of **high-dimensional multivariate data**
- Application of **regularization and dimensionality reduction**
- **Model optimization** based on quantitative performance criteria
- Clear **communication of analytical results and methodology**

---

## 🧑‍💻 Author

**Camille Auvity**  
📧 Email: [caauvity@orange.fr]  

---

