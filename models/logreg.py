"""线性回归 LinearRegression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from models.dataset import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def per_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    T2 = 0
    T0 = 0
    Tn2 = 0
    for i in range(len(y_hat)):
        y_actual[i] = round(y_actual[i])
        y_hat[i] = round(y_hat[i])
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    # tn, fp, fn, tp
    import prettytable as pt

    # tb = pt.PrettyTable( ["City name", "Area", "Population", "Annual Rainfall"])
    tb = pt.PrettyTable()
    y_train_hat = y_hat
    y_train = y_actual
    ac = pre = recall = f1 = roc = 0
    ac = round(accuracy_score(y_train_hat, y_train), 2)
    recall = round(recall_score(y_train_hat, y_train), 2)
    f1 = round(f1_score(y_train_hat, y_train), 2)
    roc = round(roc_auc_score(y_train_hat, y_train), 2)
    pre = round(precision_score(y_train_hat, y_train), 2)
    # tb.field_names = ["TN", "FP", "FN", "TP", "precision", "accuracy", "recall", "F1", "roc"]
    # tb.add_row([TN,FP, FN, TP,pre, ac, recall, f1, roc])
    tb.field_names = ["precision", "accuracy", "recall", "F1", "roc"]
    tb.add_row([pre, ac, recall, f1, roc])
    tb.border = False
    print(tb)

    return (TN, FP, FN, TP)


def report(y_train, y_train_hat):
    y_train_hat = [round(x) for x in y_train_hat]
    # tn, fp, fn, tp = confusion_matrix(y_train, y_train_hat).ravel()
    # target_names = ['-1','1']
    print(classification_report(y_train_hat, y_train))

    # print(tn, fp, fn, tp)  # 1 1 1 1
    per_measure(y_train, y_train_hat)
    # Compute ROC curve and ROC area for each class
    y_test = y_train
    y_score = y_train_hat
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([-1, 1], [-1, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def emotion_analysis():
    print(f"情感分析-逻辑回归\n{'-'*40}")

    train_X, valid_y, train_y, valid_y, train_vec, valid_vec = preprocessing()
    logreg = LogisticRegression(random_state=0)

    logreg.fit(train_vec, train_y)
    pre_valid = logreg.predict(valid_vec)
    print('正例:', sum(pre_valid == 1))
    print('负例:', sum(pre_valid == 0))

    score = accuracy_score(pre_valid, valid_y)
    print("准确率:", score)

    report(valid_y.values, pre_valid)


if __name__ == '__main__':
    emotion_analysis()
