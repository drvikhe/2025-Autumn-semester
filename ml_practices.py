import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import warnings
import platform

warnings.filterwarnings("ignore")

# 字体
system_name = platform.system()
if system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
elif system_name == 'Windows':  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------

class MLAlgorithmPractitioner:
    def __init__(self):
        print("初始化机器学习实战演练环境...")

    def run_housing_prediction(self):
        """
        项目1：房价预测 (线性回归/岭回归)
        使用 sklearn 自带的加州房价数据集
        """
        print("\n=== 项目1: 房价预测 (岭回归) ===")
        # 1. 加载数据
        housing = datasets.fetch_california_housing()
        X = housing.data
        y = housing.target

        # 2. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. 特征标准化 (对线性模型很重要)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. 模型训练 (使用岭回归 Ridge Regression)
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # 5. 预测与评估
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
        print(f"模型均方误差 (MSE): {mse:.4f}")
        print("前5个样本预测值 vs 真实值:")
        for p, t in zip(y_pred[:5], y_test[:5]):
            print(f"  预测: {p:.2f}, 真实: {t:.2f}")

    def run_iris_classification(self):
        """
        项目2：鸢尾花分类 (KNN vs 随机森林)
        对比不同算法在同一数据集上的表现
        """
        print("\n=== 项目2: 鸢尾花分类 (KNN vs 随机森林) ===")
        # 1. 加载数据
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

        # --- 模型 A: KNN ---
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_acc = knn.score(X_test, y_test)
        print(f"KNN (k=5) 准确率: {knn_acc * 100:.2f}%")

        # --- 模型 B: 随机森林 (带网格搜索调优) ---
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20]
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        rf_acc = best_rf.score(X_test, y_test)
        print(f"随机森林 (最优参数: {grid_search.best_params_}) 准确率: {rf_acc * 100:.2f}%")

        # 简单的结果对比图
        plt.figure(figsize=(8, 4))
        plt.bar(['KNN', 'Random Forest'], [knn_acc, rf_acc], color=['skyblue', 'lightgreen'])
        plt.ylim(0.8, 1.0)
        plt.title('鸢尾花分类准确率对比')
        plt.ylabel('Accuracy')
        plt.savefig('iris_model_comparison.png')
        print("模型对比图已保存为 iris_model_comparison.png")

    def run_sentiment_analysis_mock(self):
        """
        项目3：文本情感分析 (朴素贝叶斯)
        使用模拟的小型文本数据集进行演示
        """
        print("\n=== 项目3: 文本情感分析 (朴素贝叶斯) ===")

        # 1. 模拟数据 (1: 正面, 0: 负面)
        reviews = [
            "This movie is amazing and I love it",
            "Great plot and wonderful acting",
            "I really enjoyed the film",
            "Best movie ever seen",
            "Terrible movie, waste of time",
            "Boring plot and bad acting",
            "I hate this film",
            "Worst experience ever"
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]

        print(f"样本数量: {len(reviews)} 条")

        # 2. 特征提取 (词袋模型)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(reviews)

        # 3. 划分数据
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

        # 4. 训练朴素贝叶斯
        nb = MultinomialNB()
        nb.fit(X_train, y_train)

        # 5. 测试
        y_pred = nb.predict(X_test)
        print("测试集分类报告:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        # 演示预测新句子
        new_sentences = ["I love the acting", "This is boring"]
        new_X = vectorizer.transform(new_sentences)
        new_pred = nb.predict(new_X)
        print(f"新文本预测演示: {list(zip(new_sentences, ['Positive' if p == 1 else 'Negative' for p in new_pred]))}")

    def run_all(self):
        self.run_housing_prediction()
        self.run_iris_classification()
        self.run_sentiment_analysis_mock()


if __name__ == "__main__":
    practitioner = MLAlgorithmPractitioner()
    practitioner.run_all()