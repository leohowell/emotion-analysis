"""线性回归 LinearRegression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models.dataset import preprocessing


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


if __name__ == '__main__':
    emotion_analysis()
