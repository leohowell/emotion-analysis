"""数据集"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset"


def load_dataset():
    """加载评论数据集"""
    df = pd.read_csv(DATASET_PATH / 'reviews.csv')
    df['content_type'] = df['content_type'] \
        .map(lambda x: 1.0 if x == 'pos' else 0.0)
    return df


def preprocessing():
    """数据预处理"""
    df = load_dataset()
    train_X, valid_X, train_y, valid_y = \
        train_test_split(df['content'], df['content_type'],
                         test_size=0.2, random_state=42)
    model_tfidf = TFIDF(min_df=5, max_features=5000, ngram_range=(1, 3),
                        use_idf=1, smooth_idf=1)
    # 学习idf vector
    model_tfidf.fit(train_X)
    # 把文档转换成 X矩阵（该文档中该特征词出现的频次），行是文档个数，列是特征词的个数
    train_vec = model_tfidf.transform(train_X)
    valid_vec = model_tfidf.transform(valid_X)
    return train_X, valid_y, train_y, valid_y, train_vec, valid_vec
