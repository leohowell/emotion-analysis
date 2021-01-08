# 电商产品评论数据情感分析 

### 项目介绍
商业数据分析与挖掘课程作业(2020-2021)

### 代码结构
```
emotion-analysis
|- dataset (数据集)
|- download_data (数据抓取工具，词云工具)
|- models (模型)
|- main.py 入口
|- README.md
|- requirements.txt 
```
### 数据抓取工具

jd_spider.py支持JD的评论数据的抓取，会根据商品名字生成csv放在dataset中。有时候连接失败，可能需要打开京东网页再试。如果重新抓取需要删除csv文件。
商品id可以在网址中获取，支持一次性抓取多个商品。

### 词性标注工具

word_property.py支持返回DataFrame格式的数据，包含分词和词性标注等等。

### 模型结果

1. 逻辑回归 准确率 0.7
2. 决策树
3. 随机森林
4. 线性回归
5. XGboost

### 代码运行
1. 创建虚拟环境 https://docs.conda.io/en/latest/miniconda.html
2. 安装依赖 pip install -r requirements.txt
3. 执行 python main.py

```
$ python main.py
情感分析-逻辑回归
----------------------------------------
正例: 87
负例: 313
准确率: 0.7
```
