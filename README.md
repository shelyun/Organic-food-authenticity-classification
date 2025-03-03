# Organic-food-authenticity-classification
"Non-targeted Mass Spectrometry and Machine Learning Integration for Classification of Organic and Conventional Crops: A Case Study on Tomatoes"
<img width="411" alt="image" src="https://github.com/user-attachments/assets/948ca913-4b15-4204-be6f-6b49a24d5d0f" />


The scripts numbered 1-7 contain the code of the subject from data preprocessing and model training, as follows:

1_data_preprocessing.py：this script divides the boxes and adds the peak intensity values of m/z peaks in the boxes to obtain the data matrix for the number of boxes (50,200,400,600,800,1000). The corresponding files are stored in the "data" folder. The method of normalization and removal of all zero features was tried. Four feature matrices are generated (2_mz_matrix_N_windows.csv, 2_mz_matrix_N_windows_normalized.csv, 3_mz_matrix_N_windows_delete0.csv, 3_mz_matrix_N_windows_normalized_delete0.csv).

2_get_sample_list.py: obtain 160 tomato samples and corresponding labels.

3_0model_trainning.py: for model training, Using svm, xgboost, adaboost, pls, random_forest, logistics_regression, naive_bayes, knn method modeling.

3_1adaboost_importance.py: conduct feature importance analysis on the model with 1000 boxes and optimal hyperparameter combination, draw SHAP diagram, and obtain the top 20 important box number ranges.

3_2important_feature_counting.py: determine the positive or negative contribution of 20 important boxes to the predicted results, and find all the m/z peaks in each box. The frequency of m/z peaks that contribute to the organic category in organic samples and the frequency of m/z peaks that contribute to the non-organic category in non-organic samples were counted.

4_umap.py: draws the feature matrix distribution of training + verification and test sets when the number of boxes is 1000.
4_umap_raw data. py, draws the feature matrix distribution of the training + verification and test set when the number of boxes is 1000 for the original data.

5_0_0_scatterplot.py: plots the distribution of peak intensities over the number of boxes.

5_2_barplot_3D.py: AUROC Precision recall MCC comparison graph for different machine learning methods

6_compare_with_literature.py, re-establish the feature matrix in the original literature, use adaboost modeling, and view the results on the test set

7_lietrature_10fold_validation.py, tried the results of 10-fold cross-validation of the original literature method.
