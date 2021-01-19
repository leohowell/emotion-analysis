from sklearn.tree import DecisionTreeRegressor
from models.dataset import preprocessing
from models.logreg import accuracy_score, report


def emotion_analysis():
    print(f"情感分析-逻辑回归\n{'-'*40}")

    train_X, valid_y, train_y, valid_y, train_vec, valid_vec = preprocessing()

    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train_vec, train_y)
    pre_valid = tree_reg.predict(valid_vec)

    print('正例:', sum(pre_valid == 1))
    print('负例:', sum(pre_valid == 0))

    # score = accuracy_score(pre_valid, valid_y)
    # print("准确率:", score)

    report(valid_y.values, pre_valid)


if __name__ == '__main__':
    emotion_analysis()
