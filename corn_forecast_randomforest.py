import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

# 1. 加载数据
# 替换为你的数据路径
train_data = pd.read_csv('/Users/yanshiyao/PycharmProjects/PythonProject/corn_forecast/agriyield-2025/train.csv')
test_data = pd.read_csv('/Users/yanshiyao/PycharmProjects/PythonProject/corn_forecast/agriyield-2025/test.csv')

# 查看数据基本信息
print("训练集形状:", train_data.shape)
print("测试集形状:", test_data.shape)
print("\n训练集前5行:")
print(train_data.head())

# 2. 数据预处理与特征工程
# 2.1 分离特征和目标变量
# 移除无用的field_id
X = train_data.drop(columns=['field_id', 'yield'])  # 训练特征
y = train_data['yield']  # 目标变量（产量）
test_features = test_data.drop(columns=['field_id'])  # 测试特征

# 2.2 处理缺失值（使用中位数填充）
X = X.fillna(X.median())
test_features = test_features.fillna(X.median())  # 使用训练集的中位数填充测试集

# 2.3 创建交互特征
# 土壤特性交互
X['soil_ph_organic'] = X['soil_ph'] * X['organic_matter']
X['sand_rainfall'] = X['sand_pct'] * X['rainfall']

# 气候交互特征
X['temp_humidity'] = X['temperature'] * X['humidity']
X['temp_deviation_squared'] = (X['temperature'] - 25) ** 2  # 温度偏离最适值的平方

# 对测试集应用相同的特征工程
test_features['soil_ph_organic'] = test_features['soil_ph'] * test_features['organic_matter']
test_features['sand_rainfall'] = test_features['sand_pct'] * test_features['rainfall']
test_features['temp_humidity'] = test_features['temperature'] * test_features['humidity']
test_features['temp_deviation_squared'] = (test_features['temperature'] - 25) ** 2

# # 2.4 特征选择（基于皮尔逊相关系数）
# # 计算特征与产量的相关性
# corr_data = pd.concat([X, y], axis=1)
# corr_with_yield = corr_data.corr()['yield'].drop('yield')
#
# # 筛选相关系数绝对值>0.2的特征，同时保留领域知识重要特征
# selected_features = corr_with_yield[abs(corr_with_yield) > 0.2].index.tolist()
# domain_features = ['ndvi']  # NDVI是植被生长关键指标，强制保留
# for feat in domain_features:
#     if feat not in selected_features:
#         selected_features.append(feat)
#
# # 应用特征选择
# X = X[selected_features]
# test_features = test_features[selected_features]
#
# print("\n筛选后的特征:", selected_features)

# 3. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80%训练，20%验证
)

# 4. 训练随机森林模型
# 初始化模型（可根据需要调整参数）
rf = RandomForestRegressor(
    n_estimators=200,  # 树的数量
    max_depth=8,  # 树的最大深度，控制复杂度
    min_samples_split=10,  # 分裂内部节点所需的最小样本数
    random_state=42,  # 固定随机种子，确保结果可复现
    n_jobs=-1  # 使用所有可用CPU核心
)

# 训练模型
rf.fit(X_train, y_train)

# 5. 模型评估
# 在验证集上预测
y_pred = rf.predict(X_val)

# 计算RMSE
y_true_clipped = np.clip(y_val, a_min=1, a_max=None)
y_pred_clipped = np.clip(y_pred, a_min=1, a_max=None)
y_val=np.log(y_val)
y_pred=np.log(y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"\n验证集RMSE: {rmse:.2f}")

# 交叉验证（更稳健的评估）
cv_scores = cross_val_score(
    rf, X, y, cv=5,  # 5折交叉验证
    scoring='neg_mean_squared_error'  # 返回负MSE，需转换为RMSE
)
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"5折交叉验证平均RMSE: {cv_rmse:.2f}")

# 6. 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf.feature_importances_
}).sort_values(by='重要性', ascending=False)

print("\n特征重要性:")
print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['特征'], feature_importance['重要性'])
plt.xlabel('重要性分数')
plt.title('随机森林特征重要性')
plt.gca().invert_yaxis()  # 重要性高的在上方
plt.show()

# 7. 对测试集进行预测并生成提交文件
test_preds = rf.predict(test_features)

# 生成提交文件
submission = pd.DataFrame({
    'field_id': test_data['field_id'],  # 测试集的地块ID
    'yield': test_preds  # 预测的产量
})

# 保存为CSV
submission.to_csv('corn_yield_rf_submission.csv', index=False)
print("\n预测结果已保存为 corn_yield_rf_submission.csv")
