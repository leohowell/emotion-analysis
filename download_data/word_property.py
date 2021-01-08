import re

import jieba.posseg as psg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from wordcloud import WordCloud


def get_words(csv, except_words, stop_words):
    reviews = pd.read_csv(csv)
    reviews = reviews[['content', 'content_type']].drop_duplicates()
    content = reviews['content']
    print('删除数据记录中所有列值相同的记录')
    print('reviews:')
    print(reviews.head())

    str_info = re.compile(except_words)
    content = content.apply(lambda x: str_info.sub('', x))
    print('去除英文、数字、京东等词语')
    print('content:')
    print(content.head())
    # 分词
    seg_word = content.apply(lambda s: [(x.word, x.flag) for x in psg.cut(s)])  # 自定义简单分词函数
    # seg_word.head()
    print('将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置')
    print('n_content:')
    print(seg_word.head())
    n_word = seg_word.apply(lambda x: len(x))
    print('每一评论中词的个数')
    print('n_word:')
    print(n_word.head())
    n_content = [[x + 1] * y for x, y in zip(list(seg_word.index), list(n_word))]
    print('n_content:')
    print(n_content[0][:5])
    print(n_content[1][:5])

    # 将嵌套的列表展开，作为词所在评论的id
    index_content = sum(n_content, [])

    seg_word = sum(seg_word, [])
    # 词
    word = [x[0] for x in seg_word]
    # 词性
    nature = [x[1] for x in seg_word]

    content_type = [[x] * y for x, y in zip(list(reviews['content_type']), list(n_word))]
    # 评论类型
    content_type = sum(content_type, [])

    word_df = pd.DataFrame({"index_content": index_content,
                            "word": word,
                            "nature": nature,
                            "content_type": content_type})
    # word_df.head()
    print('将嵌套的列表展开，作为词所在评论的id')
    print('word_df:')
    print(word_df.head())
    # 删除标点符号
    word_df = word_df[word_df['nature'] != 'x']  # x表示标点符号
    print('删除标点符号')
    print('word_df:')
    print(word_df.head())

    # 删除停用词
    stop_path = open(stop_words, 'r', encoding='UTF-8')
    stop = stop_path.readlines()
    stop = [x.replace('\n', '') for x in stop]
    word = list(set(word) - set(stop))
    word_df = word_df[word_df['word'].isin(word)]
    # word_df.head()
    print('删除停用词')
    print('word_df:')
    print(word_df.head())

    n_word = list(word_df.groupby(by=['index_content'])['index_content'].count())
    index_word = [list(np.arange(0, y)) for y in n_word]
    print('构造各词在对应评论的位置列')
    print('index_word:')
    print(index_word[0][:5])

    index_word = sum(index_word, [])
    print('词语在该评论的位置')
    print('index_word:')
    print(index_word[:5])

    word_df['index_word'] = index_word
    print('合并评论id')
    print('word_df:')
    print(word_df.head())

    # word_df.head()
    return word_df


def show_word_cloud(df: DataFrame):
    # 提取含有名词类的评论,即词性含有“n”的评论
    ind = df[['n' in x for x in df['nature']]]['index_content'].unique()
    word_df = df[[x in ind for x in df['index_content']]]
    word_df.head()

    frequencies = df.groupby('word')['word'].count()
    frequencies = frequencies.sort_values(ascending=False)
    background_image = plt.imread('../dataset/pl.jpg')

    # 自己上传中文字体到kesci
    font_path = '../dataset/han.ttc'
    word_cloud = WordCloud(font_path=font_path,  # 设置字体，不设置就会出现乱码
                           max_words=100,
                           background_color='white',
                           mask=background_image)  # 词云形状

    my_word_cloud = word_cloud.fit_words(frequencies)
    plt.imshow(my_word_cloud)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # result = get_words('../dataset/reviews/3914278.csv', '[0-9a-zA-Z]|京东|五谷|磨房|红豆|薏米|粉|代餐粉|薏仁|红枣|杂粮|粉|',
    #                    '../dataset/stoplist.txt')
    # 返回空调单品电评论清理后DataFrame数据
    result = get_words('../dataset/reviews/100007627021.csv',
                       '[0-9a-zA-Z]|态度|感觉|物流|速度|专业|电|制冷|外形|服务|特色|产品|一级|能耗|声音|能效|师傅|效果|制冷外形|外观|'
                       'KFR|35GW|N8XHB1|挂机|空调|壁挂式|京东|冷暖效果：|静音效果：|节能省电：|送货安装：|美的|空调|安装|送货|Midea|'
                       '新一级|i青春II|智能家电|变频冷暖|1.5匹',
                       '../dataset/stoplist.txt')
    # show_word_cloud(result)  # 生成词云测试
