# Integrating Non-Targeted Mass Spectrometry and Machine Learning  
## Classification of Organic and Conventionally Grown Agricultural Products:  
### A Case Study on Tomatoes

---

## Research Methods and Routes
![Research Workflow](https://github.com/user-attachments/assets/948ca913-4b15-4204-be6f-6b49a24d5d0f)

The scripts numbered **1-7** contain the code for data preprocessing, feature extraction, and model training. Details are as follows:

---
### 1. `1_data_preprocessing.py`
- **Purpose**:  
   This script divides the boxes and adds the peak intensity values of m/z peaks in the boxes to obtain the data matrix.  
- **Features**:  
   - Supports different numbers of boxes: `50, 200, 400, 600, 800, 1000`.  
   - Normalized the data and removed all zero features.  
   - Generates four feature matrices:  
     - `2_mz_matrix_N_windows.csv`  
     - `2_mz_matrix_N_windows_normalized.csv`  
     - `3_mz_matrix_N_windows_delete0.csv`  
     - `3_mz_matrix_N_windows_normalized_delete0.csv`.  
- **Output Folder**: All results are stored in the `data` folder.

---

### 2. `2_get_sample_list.py`
- **Purpose**:  
   Obtains **160 tomato samples** and their corresponding labels.

---

### 3. `3_0model_trainning.py`
- **Purpose**:  
   Trains models using various machine learning algorithms, including:  
   - `svm`, `xgboost`, `adaboost`, `pls`, `random_forest`, `logistics_regression`, `naive_bayes`, and `knn`.

---

### 4. `3_1adaboost_importance.py`
- **Purpose**:  
   - Conducts feature importance analysis on the model with **1000 boxes** and the optimal hyperparameter combination.  
   - Draws the **SHAP diagram**.  
   - Outputs the top **20 important box number ranges**.

---

### 5. `3_2important_feature_counting.py`
- **Purpose**:  
   - Determines the **positive or negative contribution** of the 20 important boxes to the predicted results.  
   - Identifies all the m/z peaks in each box.  
   - Counts the frequency of m/z peaks that contribute to:  
     - **Organic category** in organic samples.  
     - **Non-organic category** in non-organic samples.

---

### 6. `4_umap.py`
- **Purpose**:  
   - Draws the **feature matrix distribution** of the training + validation set and test set when the number of boxes is 1000.  
   - Includes an additional script `4_umap_raw_data.py` for drawing the same distribution for the original data.

---

### 7. `5_0_0_scatterplot.py`
- **Purpose**:  
   - Plots the **distribution of peak intensities** over the number of boxes.

---

### 8. `5_2_barplot_3D.py`
- **Purpose**:  
   - Plots the **AUROC, Precision-recall, and MCC comparison graphs** for different machine learning methods.

---

### 9. `6_compare_with_literature.py`
- **Purpose**:  
   - Re-establishes the feature matrix using the method from the original literature.  
   - Uses `adaboost` for modeling and evaluates the results on the test set.

---

### 10. `7_lietrature_10fold_validation.py`
- **Purpose**:  
   - Tests the results of **10-fold cross-validation** using the method from the original literature.

---

## Repository Main Structure

```plaintext
├── code/
│   ├── 1_data_preprocessing.py
│   ├── 2_get_sample_list.py
│   ├── 3_0model_trainning.py
│   ├── 3_1adaboost_importance.py
│   ├── 3_2important_feature_counting.py
│   ├── 4_umap.py
│   ├── 4_umap_raw_data.py
│   ├── 5_0_0_scatterplot.py
│   ├── 5_2_barplot_3D.py
│   ├── 6_compare_with_literature.py
│   └── 7_lietrature_10fold_validation.py
├── data/
│   ├── 50
│   ├── 200
│   ├── 400
│   ├── 600
│   ├── 800
│   └── 1000
