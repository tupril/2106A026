# MKT3434_2025 Project Enhancements
Author: Yusuf Tugrul Demir 
Student ID: 2106A026 

## Overview
This repository contains my enhanced version of the original Python/PyQt GUI created for the MKT3434_2025 course. I have extended the application to include several additional features that make it more flexible and useful for machine learning tasks, especially for handling missing data and tuning different model parameters.

## New Features and Improvements

1) Missing Data Handling
   There is now a “Missing Data” dropdown in the “Data Management” section with these options: 
   - No Handling 
   - Mean Imputation 
   - Interpolation 
   - Forward Fill 
   - Backward Fill 

   The selected method is automatically applied before the train/test split is done, allowing cleaner data preparation.

2) SGD Regression and Classification
   Under “Classical ML,” I added “SGD Regression” and “SGD Classification” sections. For regression, you can choose MSE, MAE, or Huber loss. For classification, you can pick Cross-Entropy (log_loss) or Hinge. Both are powered by scikit-learn’s SGDRegressor and SGDClassifier.

3) Support Vector Regression (SVR)
   The “Regression” area now has “Support Vector Regression.” You can configure parameters such as C, kernel (linear, rbf, poly), degree, and epsilon.

4) **GaussianNB with Custom Priors
   In the “GaussianNB” section, I added `var_smoothing` and a line edit for custom `priors`. You can type something like “0.3,0.7” to define your own prior probabilities if you have two classes.

5) Confusion Matrix Visualization
   When you train a classification model, the GUI shows a confusion matrix and a PCA-based 2D scatter plot (if your feature dimension is greater than 2) in the Visualization panel. This helps analyze model performance in more detail.

6) Boston Housing with SVR
   If you choose the “Boston Housing Dataset” and then train with “Support Vector Regression,” you can see regression metrics such as MSE, RMSE, and R² in the Visualization panel, along with a scatter plot of actual vs. predicted values.

7) Deep Learning (MLP)
   The “Deep Learning” tab allows adding multiple layers (Dense, Dropout, etc.) and training a neural network. The training history, including accuracy or loss curves, appears in the Visualization area.

