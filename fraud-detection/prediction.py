import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def predict(train_df, model_f, cmatrix_f, auprc_f):
    # Load dataset
    df = pd.read_csv(train_df)

    # Normalize 'time' and 'amount'
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Prepare features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the model
    loaded_bst = xgb.Booster()
    loaded_bst.load_model(model_f)

    # Predictions using the loaded model
    dtest = xgb.DMatrix(X_test)
    test_preds = loaded_bst.predict(dtest)

    # Convert probabilities to binary predictions
    test_predictions = (test_preds > 0.5).astype(int)

    # Evaluate the loaded model
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    print(f'\nLoaded XGBoost Model Evaluation Metrics on Test Set:')
    print(f'Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-score: {test_f1:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_predictions)

    # Define labels for each quadrant
    labels = [
        ['True Negative\n(TN)\n{}'.format(cm[0, 0]), 'False Positive\n(FP)\n{}'.format(cm[0, 1])],
        ['False Negative\n(FN)\n{}'.format(cm[1, 0]), 'True Positive\n(TP)\n{}'.format(cm[1, 1])]
    ]

    # Plotting the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"fontsize": 12})

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Update tick labels
    plt.xticks([0.5, 1.5], ['Predicted Not Fraud (0)', 'Predicted Fraud (1)'])
    plt.yticks([0.5, 1.5], ['Actual Not Fraud (0)', 'Actual Fraud (1)'])

    plt.tight_layout()
    plt.savefig(cmatrix_f)  # Save the confusion matrix plot

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, test_preds)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")

    # Annotations for better understanding
    plt.annotate('High Precision', xy=(0.2, 0.9), xytext=(0.3, 0.95),
                 arrowprops=dict(facecolor='blue', shrink=0.05),
                 fontsize=12, color='blue')
    plt.annotate('Precision starts to fall', xy=(0.8, 0.6), xytext=(0.6, 0.7),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red')

    plt.tight_layout()
    plt.savefig(auprc_f)
    plt.close()


def main():
    datasets_dir = 'datasets'
    results_dir = 'results'

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Load datasets
    train_df = os.path.join(datasets_dir, "creditcard.csv")
    model_f = os.path.join(datasets_dir, "fraud_model.ubj")
    cmatrix_f = os.path.join(results_dir, "confusion_matrix.png")
    auprc_f = os.path.join(results_dir, "aurpc.png")

    predict(train_df, model_f, cmatrix_f, auprc_f)


if __name__ == '__main__':
    main()
