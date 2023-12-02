from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import numpy as np
def parameter(label, preds):
    C2 = confusion_matrix(label[0],preds[0])
    TP = C2[1][1]
    TN = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    acc = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    sensitivity = TP/(TP+FN+0.01)
    specificity = 1-(FP/(FP+TN+0.01))
    false_positive_rate = FP/(FP+TN+0.01)
    positive_predictive_value = TP/(TP+FP+0.01)
    negative_predictive_value = TN/(FN+TN+0.01)
    F1score = 2* precision * recall/(precision+recall)
    state = {
        'accuracy': round(acc,4),
        'precision': round(precision,4),
        'recall': round(recall,4),
        'Sensitivity': round(sensitivity,4),
        'Specificity': round(specificity,4),
        'False_positive_rate': round(false_positive_rate,4),
        'Positive_predictive_value': round(positive_predictive_value,4),
        'Negative_predictive_value':round(negative_predictive_value,4),
        'F1score': round(F1score,4)
    }
    return state


def plot_mutil_class(fpr, tpr, roc_auc,y_test,y_score,n_classes, title_name = 'Model1 of Receiver operating characteristic to multi-class',label = ['mild','modrate','serious'],colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])):
    for i in range(n_classes):
        fpr[i],tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
    lw=2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of {0} class (area = {1:0.2f})'
                ''.format(label[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_name)
    plt.legend(loc="lower right")
    plt.show()