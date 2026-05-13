#  Pima Indians Diabetes Diagnostic System
### *A High-Recall Medical Screening Tool using Machine Learning*

This project implements an optimized binary classification system to predict the onset of diabetes based on diagnostic measurements. The primary goal was to maximize **Recall** (Sensitivity) to ensure that potential diabetic patients are not missed, while maintaining a reliable level of accuracy.

---

##  Key Achievements
*   **Optimal Recall:** Achieved a peak recall of **0.89**, catching nearly 90% of diabetic cases in the test set.
*   **Precision Engineering:** Improved accuracy and reduced false positives by implementing custom interaction features.
*   **Algorithm Showdown:** Conducted a rigorous comparison between a **Tuned Random Forest** and a **Feature-Engineered Logistic Regression**.
*   **Safety-First Thresholding:** Shifted the classification threshold from **50%** to **35%** to prioritize medical screening safety.

---

##  The Data Pipeline
The model utilizes the **Pima Indians Diabetes Dataset**.

### 1. Preprocessing & Imputation
*   Identified medically impossible **0** values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI.
*   Applied **Median Imputation** to handle missing values, ensuring the model wasn't skewed by outliers.
*   Implemented **Standard Scaling** to normalize features for the Logistic Regression algorithm.

### 2. Feature Engineering (The "Super Features")
To capture compounding biological risks that a linear model might miss, two interaction features were engineered:
*   **Glucose-BMI Risk:** $Glucose \times BMI$
*   **Age-Pregnancies Risk:** $Age \times Pregnancies$
> *Adding these increased the feature count from 8 to 10 and boosted overall Accuracy by approximately 1%.*

---

##  Model Architecture

### **Model A: Tuned Random Forest**
*   **Optimization:** `GridSearchCV` with 5-fold Cross-Validation.
*   **Hyperparameters:** 300 estimators, `max_depth=5`, and `class_weight='balanced'`.
*   **Logic:** An ensemble of trees focusing on non-linear decision boundaries.

### **Model B: Engineered Logistic Regression (Champion)**
*   **Type:** Linear Classifier with weighted classes.
*   **Strategy:** Manually introduced non-linearity via feature engineering.
*   **Outcome:** Outperformed the Random Forest in both Recall and Accuracy, proving that smart data handling often beats algorithm complexity.

---

## 📈 Performance Comparison

| Metric | Random Forest (Tuned) | Logistic Regression (Engineered) |
| :--- | :--- | :--- |
| **Accuracy** | 0.69 | **0.74** |
| **Recall (Diabetes)** | 0.87 | **0.89** |
| **Precision (Diabetes)** | 0.54 | **0.59** |
| **Threshold** | 35% | **35%** |

---

##  Installation & Usage

### Prerequisites
*   Python 3.x
*   Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn

### Running the Diagnostic
1.  Clone the repository.
2.  Open the Jupyter/Colab notebook.
3.  Run the **Multi-Model Diagnostic Report** cell to input new patient data and receive a risk probability from both models.

```python
# Example Usage:
new_patient = {
    'Pregnancies': [1],
    'Glucose': [120],
    'BloodPressure': [70],
    'SkinThickness': [20],
    'Insulin': [80],
    'BMI': [30.5],
    'DiabetesPedigree': [0.45],
    'Age': [35]
}
# The system will automatically engineer features and apply the 35% threshold.
