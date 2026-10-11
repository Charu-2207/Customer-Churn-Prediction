# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn occurs when a customer stops using a company's service.
Predicting churn can help businesses identify customers who may be at
risk of leaving and make better-informed retention decisions.

This project explores customer churn prediction using the **Telco
Customer Churn dataset**. It includes data exploration, preprocessing,
model training, and evaluation using machine learning algorithms.

## Dataset

-   **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
-   **Target column:** `Churn`
-   **Task:** Binary classification --- predict whether a customer
    churns (`Yes`) or stays (`No`).
-   **Dataset size:** 7,043 rows and 21 columns.

The dataset contains customer information and service/account details.
Check the dataset's license and redistribution terms before sharing the
CSV publicly in a repository.

## Models Explored

The notebook compares the following classification algorithms:

-   Decision Tree
-   Random Forest
-   XGBoost

The notebook also explores class-imbalance handling with **SMOTE** and
uses **five-fold cross-validation** to compare model performance.

## Workflow

1.  Load the dataset.
2.  Explore the data and examine the target distribution.
3.  Prepare the data for machine learning, including encoding
    categorical features.
4.  Split the data into training and testing sets.
5.  Apply SMOTE to address class imbalance in the training workflow.
6.  Train and compare the selected classification models.
7.  Evaluate model performance using classification metrics.
8.  Save the trained model and encoders for later predictions.
9.  Run a sample prediction for a customer record.

> **Note:** Make sure SMOTE is applied only to the training data, not
> before the train/test split, to avoid data leakage.

## Technologies Used

-   Python
-   Jupyter Notebook / Google Colab
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   Imbalanced-learn (SMOTE)

## Repository Structure

``` text
customer-churn-prediction/
├── Cust_Seg_Chrn_Pred.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

If you do not include the dataset in the repository, add instructions in
this README explaining where to obtain it and where to place the CSV
file.

## How to Run

### Option 1: Google Colab

1.  Upload `Cust_Seg_Chrn_Pred.ipynb` to Google Colab.
2.  Upload the dataset, or mount Google Drive and update the dataset
    path in the notebook.
3.  Install any missing dependencies in a notebook cell.
4.  Run the notebook cells in order.

### Option 2: Run Locally

1.  Clone or download this repository.

2.  Install Python and Jupyter Notebook.

3.  Install the libraries used by the notebook:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn notebook
    ```

4.  Place the CSV file in the path expected by the notebook, or update
    the path in the data-loading cell.

5.  Start Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

6.  Open `Cust_Seg_Chrn_Pred.ipynb` and run the cells in order.

## Evaluation

The notebook explores cross-validation accuracy for the models. The
results recorded during development were approximately:

  Model             Mean cross-validation accuracy
  --------------- --------------------------------
  Decision Tree                                78%
  Random Forest                                84%
  XGBoost                                      83%

These values are notebook-reported results and may vary with the data
split, preprocessing, random seed, and library versions. Re-run the
notebook to reproduce them. Before publishing a final test score, verify
that the evaluation cell explicitly evaluates the intended model; do not
rely on a variable left over from a model-training loop.

Accuracy alone may not fully describe churn performance. Review
precision, recall, F1-score, and the confusion matrix as well,
especially because the classes may be imbalanced.

## Output

The notebook includes a sample customer prediction with predicted churn
probabilities. It also saves model-related files, such as:

-   `customer_churn_model.pkl`
-   `encoders.pkl`

These files are generated outputs; they may need to be recreated by
running the notebook. Only commit model artifacts if you intend to share
them and they do not make the repository unnecessarily large.

## Future Improvements

-   Correctly select and evaluate the final model in a dedicated
    evaluation cell.
-   Tune hyperparameters and compare precision, recall, F1-score, and
    ROC-AUC.
-   Use a reproducible train/test split and document random seeds.
-   Add visualizations of feature importance and churn patterns.
-   Build a simple interface for entering customer details and viewing a
    prediction.

## Disclaimer

This project is for educational and learning purposes. Predictions are
estimates and should not be treated as guaranteed outcomes.

