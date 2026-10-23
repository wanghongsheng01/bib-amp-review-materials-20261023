
import numpy as np
from sklearn import mixture
from sklearn.model_selection import train_test_split


class PeptideActivityPredictor:
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.gmm_0 = mixture.GaussianMixture(n_components=n_components,
                                             covariance_type=covariance_type,
                                             random_state=random_state)
        self.gmm_1 = mixture.GaussianMixture(n_components=n_components,
                                             covariance_type=covariance_type,
                                             random_state=random_state)

    def fit(self, X, y):
        # 将数据集分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 分别用相应类别的数据拟合模型
        self.gmm_0.fit(X_train[y_train == 0])
        self.gmm_1.fit(X_train[y_train == 1])

        # 保存测试集，以便以后进行验证
        self.X_test = X_test
        self.y_test = y_test

    def predict_proba(self, X):
        # 计算在两个GMM模型上的得分（负对数似然）
        log_likelihood_0 = self.gmm_0.score_samples(X)
        log_likelihood_1 = self.gmm_1.score_samples(X)

        # 将对数似然转换为概率
        prob_0 = np.exp(log_likelihood_0)
        prob_1 = np.exp(log_likelihood_1)

        # 标准化以获得活性值的分布
        prob_total = prob_0 + prob_1
        activity_distribution = prob_1 / prob_total

        return activity_distribution

    def predict(X, gmm_0, gmm_1):
        scores_0 = gmm_0.score_samples(X)
        scores_1 = gmm_1.score_samples(X)
        preds = (scores_1 > scores_0).astype(int)
        return preds


# 使用示例
# 假设我们有以下数据
num_samples = 100
num_features = 3
X = np.random.rand(num_samples, num_features)
y = np.random.choice([0, 1], size=num_samples)

# 创建预测器实例
predictor = PeptideActivityPredictor()

# 拟合模型
predictor.fit(X, y)

# 假设 new_X 是新的化学描述符数据
new_X = np.random.rand(5, num_features)  # 生成一些新的数据用于推理

# 使用模型进行推理，得到活性分布
activity_distribution = predictor.predict_proba(new_X)
print("Activity distribution for new samples:", activity_distribution)
