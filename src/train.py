import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime


from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, \
    mean_absolute_percentage_error
import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1.定义电力负荷模型类，配置日志，获取数据源
class PowerLoadModel():
    # 1.1 初始化属性信息
    def __init__(self):
        # 1。2 拼接日志文件名
        logfile_name = "train" + pd.to_datetime(datetime.datetime.now()).strftime("%Y%m%d%H%M%S")
        # 1.3 创建日志对象
        self.logger = Logger("../", logfile_name).get_logger()
        # 1.4 测试写一条日志
        self.logger.info("开始创建电力负荷模型对象啦")
        # 1.5 获取数据源
        self.data_source = data_preprocessing("../data/train.csv")


# 2.查看数据整体分布情况
def ana_data(data):
    """
    1.查看数据整体分布情况
    2.负荷整体的分布情况
    3.各小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月的平均负荷趋势，看一下负荷在一天中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data:
    :return:
    """
    # 2.1查看数据整体分布情况
    ana_data = data.copy()         #防止修改原始数据
    ana_data.info()
    # 2.2 负荷整体的分布情况，直方图
    fig,axes = plt.subplots(4, 1, figsize=(20,40))
    axes[0].hist(ana_data['power_load'], bins=100)
    axes[0].set_title("负荷整体分布情况")
    axes[0].set_xlabel("负荷")

    # 2.3 各小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 2.3.1 新增一个小时列
    ana_data["hour"] = ana_data["time"].str[11:13]
    # 2.3.2 根据小时分组，计算平均负荷
    hour_ana_data = ana_data.groupby("hour", as_index=False)["power_load"].mean()
    axes[1].plot(hour_ana_data["hour"], hour_ana_data["power_load"])
    axes[1].set_title("各小时的平均负荷趋势图")
    axes[1].set_xlabel("小时")
    axes[1].set_ylabel("平均负荷")
    # 2.4 各个月的平均负荷趋势，看一下负荷在一天中的变化情况
    # 2.4.1新增一个月份列
    ana_data["month"] = ana_data["time"].str[5:7]
    # 2.4.2 根据月份分组，计算平均负荷
    month_ana_data = ana_data.groupby("month", as_index=False)["power_load"].mean()
    axes[2].plot(month_ana_data["month"], month_ana_data["power_load"])
    axes[2].set_title("各个月的平均负荷趋势图")
    axes[2].set_xlabel("月份")
    axes[2].set_ylabel("平均负荷")
    # 2.5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    # 2.5.1 新增一个工作日列
    ana_data["weekday"] = ana_data["time"].apply(lambda x: pd.to_datetime(x).weekday())
    ana_data["is_holiday"] = ana_data["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    work_holiday_load_mean = ana_data.groupby("is_holiday", as_index=False)["power_load"].mean()
    axes[3].bar(x = ["工作日","周末"], height = work_holiday_load_mean["power_load"])
    axes[3].set_title("工作日与周末的平均负荷情况")
    axes[3].set_xlabel("工作日")
    axes[3].set_ylabel("平均负荷")


    plt.tight_layout()
    plt.savefig("../data/fig/负荷整体的分布情况.png",bbox_inches='tight')
    plt.show()

# 3.特种工程
def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1.提取出时间特征：月份、小时
    2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    3.提取昨日同时刻负荷特征
    4.剔除出现空值的样本
    5.整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    logger.info("==========开始进行特征工程处理==========")
    feature_data = data.copy()
    # 1.提取出时间特征：月份、小时
    feature_data["hour"] = feature_data["time"].str[11:13]
    feature_data["month"] = feature_data["time"].str[5:7]
    # 1.2 对时间特征进行one-hot编码
    hour_data = pd.get_dummies(feature_data["hour"], prefix="hour")
    month_data = pd.get_dummies(feature_data["month"], prefix="month")
    # 1.3 拼接时间特征
    feature_data = pd.concat([feature_data, hour_data, month_data], axis=1)

    # 2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    logger.info("==========提取出相近时间窗口中的负荷特征==========")
    for i in (1,2,3):
        feature_data[f"前{i}小时负荷"] = feature_data["power_load"].shift(i)

    # 3.提取昨日同时刻负荷特征
    logger.info("==========提取昨日同时刻负荷特征==========")
    # feature_data["time"] = pd.to_datetime(feature_data["time"])
    # feature_data = feature_data.set_index("time")
    # feature_data["yesterday_load"] = feature_data["power_load"].shift(1, freq='D')
    # feature_data = feature_data.reset_index()
    # 3.1 创建一个  时间：负荷  字典
    time_load_dict = feature_data.set_index("time")["power_load"].to_dict()
    # 3.2 创建一个新列，表示昨日同时刻
    feature_data["yesterday_time"] = feature_data["time"].apply(lambda x:(pd.to_datetime(x) - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"))
    # 3.3创建一个新列，表示昨日同时刻的负荷
    feature_data["yesterday_load"] = feature_data["yesterday_time"].map(time_load_dict)
    # 3.4 删除空值
    feature_data = feature_data.dropna()
    # 3.5 整理时间特征
    feature_columns = list(hour_data.columns) + list(month_data.columns) + ["前1小时负荷", "前2小时负荷", "前3小时负荷", "yesterday_load"]
    # 3.6 返回结果
    return feature_data, feature_columns
# 4.模型训练，评估
def model_train(data, features, logger):
    """
    1.数据集切分
    2.网格化搜索与交叉验证
    3.模型实例化
    4.模型训练
    5.模型评价
    6.模型保存
    :param data: 特征工程处理后的输入数据
    :param features: 特征名称
    :param logger: 日志对象
    :return:
    """

    # 1.数据集切分
    x = data[features]
    y = data["power_load"]

    x_train, x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    # # 2.网格化搜索与交叉验证
    # logger.info("==========进行网格化搜索与交叉验证，寻找最优超参==========")
    # logger.info(f"开始时间：{datetime.datetime.now()}")
    # # 2.1参数准备
    # param_dict = {
    #     "n_estimators": [50, 100, 200],
    #     "max_depth": [3, 5, 7],
    #     "learning_rate": [0.1, 0.01]
    # }
    # # 2.2创建网格化搜索对象
    # estimator = GridSearchCV(XGBRegressor(), param_grid=param_dict, cv=5)
    # # 2.3模型训练
    # estimator.fit(x_train,y_train)
    # # 2.4 打印最优模型
    # logger.info(f"结束时间：{datetime.datetime.now()}")
    # logger.info(f"最优参数：{estimator.best_params_}")


    # 3.模型实例化
    logger.info("==========模型实例化==========")
    # 3.1创建XGBoost模型对象
    estimator = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1
    )
    # 3.2 模型训练
    estimator.fit(x_train, y_train)
    # 3.3 模型预测
    y_pred = estimator.predict(x_test)
    # 3.4 模型评估

    print(f"均方误差: {mean_squared_error(y_test, y_pred)}")
    print(f"均方根误差: {root_mean_squared_error(y_test, y_pred)}")
    print(f"平均绝对值误差: {mean_absolute_error(y_test, y_pred)}")
    print(f"平均绝对百分比误差: {mean_absolute_percentage_error(y_test, y_pred)}")
    # 3.5 模型保存
    joblib.dump(estimator, "../model/xgb_20261122.pkl")
# 5.测试
if __name__ == '__main__':
    # 5.1 创建电力负荷模型对象
    plm = PowerLoadModel()
    # print(plm.data_source)
    # 5.2 调用查看数据整体分布情况
    # ana_data(plm.data_source)
    # 5.3 特征工程
    feature_data, feature_columns = feature_engineering(plm.data_source, plm.logger)
    # 5.4 模型训练，预测
    model_train(feature_data, feature_columns, plm.logger)
