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

Extended GUI for MKT3434_2025

This project aimed to expand a pre-existing machine learning GUI by adding several important tools and methods used in real-world data analysis. These additions focused primarily on dimensionality reduction, model validation, and clustering performance evaluation. The updates provide a more complete environment for exploring high-dimensional data, testing models, and drawing insights from various datasets.
How to Run the Application
To launch the GUI, use the following command in your terminal after installing dependencies:
    python 2106A026_fully_extended.py

After the application opens, go to the tab labeled 'DimRed & CV'. From there, you can interactively apply each method described below.
Features and Their Usage

1. PCA (Principal Component Analysis)
Clicking 'Run PCA' will apply PCA to the loaded dataset and display a cumulative explained variance curve. This curve helps users understand how many components are necessary to preserve most of the information in the data. Reducing dimensionality this way can improve computation speed and reduce noise, especially before training ML models.

2. LDA (Linear Discriminant Analysis)
Clicking 'Run LDA' applies LDA, which is a supervised dimensionality reduction technique. It uses class labels to find directions that best separate the classes. The result is shown in a 2D scatter plot. This can be very useful for visualizing how well different classes are distinguishable in feature space.

3. KMeans Clustering + Elbow Method
Clicking 'KMeans Elbow' performs clustering with various numbers of clusters (k=1 to 9) and plots the distortion (inertia). The point where the curve 'bends' is the optimal cluster number. This helps users determine how many natural groupings exist in their dataset.

4. t-SNE and UMAP
These are nonlinear dimensionality reduction methods useful for visualizing complex datasets. t-SNE emphasizes local structure and shows how data points form clusters. UMAP does similar visualization but often runs faster and gives better-defined shapes. Both project high-dimensional data into 2D space so that users can understand relationships more clearly.

5. K-Fold Cross-Validation
When you click 'Run K-Fold CV', the application divides your dataset into 5 folds and trains the model on different combinations of those folds. It reports the mean accuracy and standard deviation, giving you a sense of how stable the model is. This prevents overfitting and gives a better idea of how well the model generalizes to unseen data.

6. Silhouette Score
Clicking 'Silhouette Score' calculates how well samples are clustered for k=2 to 10. Higher silhouette scores mean better-separated and tighter clusters. This tool helps users choose the most meaningful number of clusters when doing unsupervised learning.

7. Train/Validation/Test Split
Internally, every dataset is divided as 70% training, 15% validation, and 15% test. Although this split isn't directly exposed through the GUI, it’s embedded in the backend logic and ensures good generalization. This setup supports experiments with both cross-validation and single-pass training routines.

8. Dataset Loading
Users can either load their own dataset using the 'Load Dataset' button or rely on built-in datasets (like Iris) that auto-load when no data is provided. The GUI supports .csv format files. Loading the right dataset is essential before applying any method; otherwise, the system will prompt you to do so.
Conclusion
This README is meant to guide anyone using the GUI and explain not only what each button does, but also why each method is useful. The application is now more powerful and closer to what a real analyst or data scientist might use for initial exploratory work.
Quick Usage Flow – How to Use the Application
When the GUI first launches, you must load a dataset before using any of the buttons. You can either:
- Select one of the built-in datasets (like Iris, Wine, or Digits) from the dropdown menu on the left,
- Or use the 'Load Dataset' button to upload your own .csv file from your computer.
Once the dataset is loaded, the system will automatically divide it into training, validation, and test sets (70/15/15).
After loading data, go to the 'DimRed & CV' tab. There, you’ll see buttons for each available analysis method.
- 'Silhouette Score' calculates and plots a silhouette score graph for k = 2 to 10.
- 'Run K-Fold CV' uses 5-fold validation and shows the mean and standard deviation of accuracy.
These tools allow you to evaluate how well clustering or classification is performing based on the dataset you provided or selected.
In a typical workflow, you load the Iris dataset, run PCA to see explained variance, then try LDA or t-SNE to visualize group separations. After that, you might check clustering quality using silhouette score, and validate model stability with cross-validation.


