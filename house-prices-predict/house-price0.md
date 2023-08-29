# house-prices0
输入数据：波斯顿房价，多个属性
0. 数据的查看
1. 两个属性的散点图，箱型图(包括)
2. 属性的分布图，正态分布图。
3. 多个属性的热力图和选择出最相近的前n个属性的热力图。
4. 缺失数据查看，多少数据有百分之多少的缺失值。
5. 对非正太分布数据取log得到正态分布。
6. 对虚拟变量转化为多个属性，gengder-->gender_male gender_female.

# house-prices1---LASSO/Regression /Elastic Net Regression/Kernel Ridge Regression/Gradient Boosting Regression/XGBoost/LightGBM
1. 丢弃某一列的数据
train.drop("Id", axis = 1, inplace = True)
2. 对模型进行正太分布分析
sns.distplot(train['SalePrice'] , fit=norm);
res = stats.probplot(train['SalePrice'], plot=plt)
3. 根据偏度skew，对非正太分布数据取log+1或box1p得到正态分布。
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

train["SalePrice"] = np.log1p(train["SalePrice"])
4. 合并train和test数据
all_data = pd.concat((train, test)).reset_index(drop=True)
5. 将na数据转为None
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
6. 将非确定的数值型数据转化为类别，如地区号'234'-->2
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data[c] = lbl.transform(list(all_data[c].values))
7. 元学习集成多个模型并且将数据集划分进行训练
LASSO  Regression /Elastic Net Regression/Kernel Ridge Regression/Gradient Boosting Regression/XGBoost/LightGBM
8. 对预测的值恢复正常
np.expm1(model_xgb.predict(train[0:2]))

# house-prices2---Lasso/XGboost
1. 利用lasso回归，得到最相关的特征
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
2. 绘制相关系相关图
imp_coef.plot(kind = "barh")
3. 绘制预测--损失的散点图

4. XGbost方法以及其训练迭代损失图的绘制

5. 用数组画散点图
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")






