import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):
    # Put your code
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions)
    
    cm = metrics.confusion_matrix(y_test, predictions)
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features):
    # Put your code
    predicted_proba = model.predict_proba(features)
    predicted_proba = predicted_proba[:, predicted_proba.shape[1]-1]
    fpr = metrics.roc_curve(y_test, predicted_proba)[0]
    tpr = metrics.roc_curve(y_test, predicted_proba)[1]
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc