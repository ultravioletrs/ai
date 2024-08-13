import argparse
import os

import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_model(train_df, model_f):
    df = pd.read_csv(train_df)

    print("Starting training")
    # Normalize 'time' and 'amount'
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Split data into train, validation, and test sets
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Apply advanced sampling technique (SMOTETomek) to handle class imbalance on training set
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

    # Train and evaluate XGBoost model
    dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # Choose a single evaluation metric
        'eta': 0.1,  # Learning rate
        'max_depth': 6,  # Maximum depth of a tree
        'subsample': 0.8,  # Subsample ratio of the training instance
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'scale_pos_weight': len(y_train_res) / sum(y_train_res)  # Balancing the positive class
    }

    # Train the XGBoost model
    num_rounds = 500  # Number of boosting rounds (adjust as needed)
    evals = [(dval, 'validation')]
    bst = xgb.train(params, dtrain, num_rounds, evals=evals, early_stopping_rounds=10, verbose_eval=True)

    # Save the model
    bst.save_model(model_f)

    # Predictions on validation and test sets
    val_preds = bst.predict(dval)
    test_preds = bst.predict(dtest)

    # Convert probabilities to binary predictions
    val_predictions_xgb = (val_preds > 0.5).astype(int)
    test_predictions_xgb = (test_preds > 0.5).astype(int)

    # Evaluate XGBoost model
    val_accuracy_xgb = accuracy_score(y_val, val_predictions_xgb)
    val_precision_xgb = precision_score(y_val, val_predictions_xgb)
    val_recall_xgb = recall_score(y_val, val_predictions_xgb)
    val_f1_xgb = f1_score(y_val, val_predictions_xgb)

    test_accuracy_xgb = accuracy_score(y_test, test_predictions_xgb)
    test_precision_xgb = precision_score(y_test, test_predictions_xgb)
    test_recall_xgb = recall_score(y_test, test_predictions_xgb)
    test_f1_xgb = f1_score(y_test, test_predictions_xgb)

    print(f'\nXGBoost Evaluation Metrics on Validation Set:')
    print(f'Accuracy: {val_accuracy_xgb:.4f}, Precision: {val_precision_xgb:.4f}, Recall: {val_recall_xgb:.4f}, F1-score: {val_f1_xgb:.4f}')

    print(f'\nXGBoost Evaluation Metrics on Test Set:')
    print(f'Accuracy: {test_accuracy_xgb:.4f}, Precision: {test_precision_xgb:.4f}, Recall: {test_recall_xgb:.4f}, F1-score: {test_f1_xgb:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', nargs='+', help='Paths to fraud detection datasets')
    parser.add_argument('--model', default='fraud_model.ubj', help='Filename to save the trained model')
    args = parser.parse_args()

    datasets_dir = 'datasets'
    results_dir = 'results'

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    # Load datasets
    train_df = os.path.join(datasets_dir, args.data_paths[1])
    model_f = os.path.join(results_dir, args.model)

    train_and_evaluate_model(train_df, model_f)


if __name__ == '__main__':
    main()
