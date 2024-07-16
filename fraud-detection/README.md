# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques, specifically the XGBoost model. The dataset used in this project was downloaded from Kaggle.
Given the class imbalance ratio, script measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC).

## Dataset

The dataset used in this project contains transactions made by European cardholders over a period of two days in September 2013. It has been modified using Principal Component Analysis (PCA) for confidentiality reasons. 
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
The dataset includes the following features:

- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1, V2, ..., V28`: Principal components obtained from PCA.
- `Amount`: Transaction amount.
- `Class`: Label indicating whether the transaction is fraudulent (1) or not (0).

You can download the dataset `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Train Model

To train model run script execute:
```bash
python fraud-detection.py
``` 
Script will produce model `fraud_model.ubj`
### Test Model
Inference can be done using `prediction.py`

```bash
python prediction.py
```
Results will be `.png` images that represents confusion matrix and AUPRC
![](images/AUPRC.png)
![](images/C-matrix.png)







