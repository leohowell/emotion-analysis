from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF  # 原始文本转化为tf-idf的特征矩阵
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#print(pre_valid)
#print("---------")
#print(valid_y)
#print(valid_y.values)
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split           #划分训练/测试集
from sklearn.feature_extraction.text import CountVectorizer    #抽取特征
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# 将有标签的数据集划分成训练集和测试集
train_X,valid_X,train_y,valid_y = train_test_split(reviews['content'],reviews['content_type'],test_size=0.2,random_state=42)
def print_data():
    print("x_train")
    print(train_X)
    print("x_test")
    print(valid_X)
    print("train_y")
    print(train_y)
    print("void_y")
    print(valid_y)
#print_data()
train_X.shape,train_y.shape,valid_X.shape,valid_y.shape

# 模型构建
model_tfidf = TFIDF(min_df=5, max_features=5000, ngram_range=(1,3), use_idf=1, smooth_idf=1)
# 学习idf vector
model_tfidf.fit(train_X)
# 把文档转换成 X矩阵（该文档中该特征词出现的频次），行是文档个数，列是特征词的个数
train_vec = model_tfidf.transform(train_X)
train_vec.toarray()
# 模型训练
model_SVC = LinearSVC()
clf = CalibratedClassifierCV(model_SVC)
clf.fit(train_vec,train_y)
# 把文档转换成矩阵
valid_vec = model_tfidf.transform(valid_X)
# 验证
pre_valid = clf.predict_proba(valid_vec)
pre_valid[:5]
pre_valid = clf.predict(valid_vec)
print('正例:',sum(pre_valid == 1))
print('负例:',sum(pre_valid == 0))
print(pre_valid)

def per_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
        
    T2 = 0
    T0 = 0
    Tn2 = 0
    for i in range(len(y_hat)): 
        y_actual[i]= round(y_actual[i])
        y_hat[i] = round(y_hat[i])
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    #tn, fp, fn, tp
    import prettytable as pt

    # tb = pt.PrettyTable( ["City name", "Area", "Population", "Annual Rainfall"])
    tb = pt.PrettyTable()
    y_train_hat = y_hat
    y_train = y_actual
    ac=pre=recall=f1=roc=0
    ac = round(accuracy_score(y_train_hat,y_train),2)
    recall = round(recall_score(y_train_hat,y_train),2)
    f1 = round(f1_score(y_train_hat,y_train),2)
    roc = round(roc_auc_score(y_train_hat,y_train),2)
    pre = round(precision_score(y_train_hat, y_train),2)
    #tb.field_names = ["TN", "FP", "FN", "TP", "precision", "accuracy", "recall", "F1", "roc"]
    #tb.add_row([TN,FP, FN, TP,pre, ac, recall, f1, roc])
    tb.field_names = ["precision", "accuracy", "recall", "F1", "roc"]
    tb.add_row([pre, ac, recall, f1, roc])
    tb.border=False
    print(tb)
    
    
    return(TN, FP, FN, TP)

def report(y_train, y_train_hat):
    from sklearn.metrics import classification_report
    y_train_hat=[round(x) for x in y_train_hat]
    #tn, fp, fn, tp = confusion_matrix(y_train, y_train_hat).ravel()
    #target_names = ['-1','1']
    print(classification_report(y_train_hat,y_train))
    
    #print(tn, fp, fn, tp)  # 1 1 1 1
    per_measure(y_train,y_train_hat)
    # Compute ROC curve and ROC area for each class
    y_test=y_train
    y_score=y_train_hat
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

report(valid_y.values, pre_valid)
score = accuracy_score(pre_valid,valid_y)
print("准确率:",score)



#朴素贝叶斯
NB = MultinomialNB()
clf = CalibratedClassifierCV(NB)
train_vec = model_tfidf.transform(train_X)
clf.fit(train_vec,train_y)
valid_vec = model_tfidf.transform(valid_X)
pre_valid = clf.predict(valid_vec)
report(valid_y.values, pre_valid)
score = accuracy_score(pre_valid,valid_y)
print("准确率:",score)

#LDA 模型
import re
import itertools

from gensim import corpora, models


# 载入情感分析后的数据
posdata = pd.read_csv("./posdata.csv", encoding = 'utf-8')
negdata = pd.read_csv("./negdata.csv", encoding = 'utf-8')


# 建立词典
pos_dict = corpora.Dictionary([[i] for i in posdata['word']])  # 正面
neg_dict = corpora.Dictionary([[i] for i in negdata['word']])  # 负面

# 建立语料库
pos_corpus = [pos_dict.doc2bow(j) for j in [[i] for i in posdata['word']]]  # 正面
neg_corpus = [neg_dict.doc2bow(j) for j in [[i] for i in negdata['word']]]   # 负面

# 余弦相似度函数
def cos(vector1, vector2):
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1, vector2): 
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return(None)  
    else:  
        return(dot_product / ((normA*normB)**0.5))   

# 主题数寻优
def lda_k(x_corpus, x_dict):  
    
    # 初始化平均余弦相似度
    mean_similarity = []
    mean_similarity.append(1)
    
    # 循环生成主题并计算主题间相似度
    for i in np.arange(2,11):
        # LDA模型训练
        lda = models.LdaModel(x_corpus, num_topics = i, id2word = x_dict)
        for j in np.arange(i):
            term = lda.show_topics(num_words = 50)
            
        # 提取各主题词
        top_word = []
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"',i)) \
                             for i in term[k][1].split('+')])  # 列出所有词
           
        # 构造词频向量
        word = sum(top_word,[])  # 列出所有的词   
        unique_word = set(word)  # 去除重复的词
        
        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))  
            
        p = list(itertools.permutations(list(np.arange(i)),2))
        l = len(p)
        top_similarity = [0]
        for w in np.arange(l):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))
            
        # 计算平均余弦相似度
        mean_similarity.append(sum(top_similarity)/l)
    return(mean_similarity)


# 计算主题平均余弦相似度
pos_k = lda_k(pos_corpus, pos_dict)
neg_k = lda_k(neg_corpus, neg_dict)

# 绘制主题平均余弦相似度图形
from matplotlib.font_manager import FontProperties  
font = FontProperties(size=14)


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
ax1.plot(pos_k)
ax1.set_xlabel('正面评论LDA主题数寻优', fontproperties=font)

ax2 = fig.add_subplot(212)
ax2.plot(neg_k)
ax2.set_xlabel('负面评论LDA主题数寻优', fontproperties=font)

# LDA主题分析
pos_lda = models.LdaModel(pos_corpus, num_topics = 3, id2word = pos_dict)  
neg_lda = models.LdaModel(neg_corpus, num_topics = 3, id2word = neg_dict)

pos_lda.print_topics(num_words = 10)
neg_lda.print_topics(num_words = 10)

import pyLDAvis
from pyLDAvis import gensim
vis = pyLDAvis.gensim.prepare(pos_lda,pos_corpus,pos_dict)
# 需要的三个参数都可以从硬盘读取的，前面已经存储下来了

# 在浏览器中心打开一个界面
#pyLDAvis.show(vis)

# 在notebook的output cell中显示
#pyLDAvis.display(vis)
pyLDAvis.enable_notebook(vis)

