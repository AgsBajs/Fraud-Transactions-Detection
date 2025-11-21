# Credit Card Fraud Detection (Machine Learning Final Project)

This project is about using machine learning to detect **fraudulent credit card transactions** based on real-world style transaction data.

---

## What This Project Does

- Treats fraud detection as a **binary classification** problem  
  - `0` = normal transaction  
  - `1` = fraud  
- Trains and compares two models:
  - **Logistic Regression** (simple, linear baseline)
  - **Random Forest Classifier** (more powerful, non-linear model)
- Handles a **heavily imbalanced dataset** where fraud is rare
- Evaluates models with metrics that actually matter for fraud:
  - Precision, Recall, F1-score
  - ROC–AUC
  - Precision–Recall AUC

---

## How It Works (High Level)

1. **Load the dataset** of credit card transactions  
2. **Clean and preprocess the data**  
   - Turn timestamps into features like hour of day and day of week  
   - Drop ID-like columns that don’t really help (and can cause overfitting)  
   - Scale numeric features and one-hot encode categorical features  
3. **Train/test split** the data while keeping the fraud ratio similar in both sets  
4. **Build pipelines** for:
   - Logistic Regression  
   - Random Forest  
5. **Train, predict, and evaluate** each model using classification metrics, confusion matrices, and ROC/PR curves

---

## What I Learned

- **Accuracy can lie** on imbalanced data  
  - I saw that a model can get very high accuracy by mostly predicting “not fraud,” so accuracy alone isn’t enough.
- **Recall vs Precision tradeoff matters**  
  - Logistic Regression can catch a lot of fraud (high recall) but also flags many normal transactions (low precision).  
  - Random Forest is better at balancing both: fewer false alarms while still catching fraud.
- **Better metrics for imbalance**  
  - ROC–AUC and especially **Precision–Recall AUC** gave me a much clearer picture of performance when the positive class is rare.
- **Pipelines keep things clean**  
  - Using `ColumnTransformer` + `Pipeline` in scikit-learn helped me organize preprocessing and modeling in one place without a lot of manual steps.
- **A very high accuracy doesn’t always mean overfitting**  
  - On highly imbalanced datasets, I learned that you can have ~0.99 accuracy and still be generalizing well if your ROC/PR curves and other metrics also look good.

Overall: I learned how to go beyond “just accuracy,” how to think about fraud detection in a more realistic way, and how to structure an end-to-end machine learning workflow.
