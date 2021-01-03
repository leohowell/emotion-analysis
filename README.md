# 电商产品评论数据情感分析 

### 项目介绍
商业数据分析与挖掘课程作业(2020-2021)

### 模型结果

1. 逻辑回归 准确率 0.7
2. 决策树
3. 随机森林
4. 线性回归
5. XGboost

### 代码结构
```
emotion-analysis
|- dataset (数据集)
|- models (模型)
|- main.py 入口
|- README.md
|- requirements.txt 
```

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
