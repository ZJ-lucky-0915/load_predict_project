import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime


from utils.log import Logger
from utils.common import data_preprocessing
from sklearn.metrics import mean_absolute_error

import joblib

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1.配置电力负荷预测类
class PowerLoadPredict(object):
    def __init__(self):
        # 配置日志记录
        logfile_name = "predict_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.logger = Logger("../" ,logfile_name).get_logger()
        # 获取数据源
        self.data = data_preprocessing("../data/test.csv")
        # 历史数据转换为字典，key：时间，valuse：负荷，目的是为了避免频繁操作dataframe，提高效率。实际开发场景中可以使用redis缓存
        self.time_load_dict = self.data.set_index("time")["power_load"].to_dict()

# 2.特征解析
def feature_extract(data_dict,time,logger):
    """
    根据预测时间,历史负荷, 加工出特征数据
    :param time: 预测时间
    :param time_load_dict: 预测时间以前的历史负荷
    :return:
    """
    logger.info(f"==========解析时间为：{time}所对应的特征==========")
    # 特征列清单
    feature_cols = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08',
                    'hour_09',
                    'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
                    'hour_19',
                    'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_01', 'month_02', 'month_03', 'month_04',
                    'month_05', 'month_06',
                    'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12', '前1小时负荷',
                    '前2小时负荷',
                    '前3小时负荷', 'yesterday_load']
    # 1.解析时间特征
    # 1.1 截取要预测的time字段的 小时信息
    pre_hour = time[11:13]
    hour_list = []
    for i in range(24):
        if pre_hour == feature_cols[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    # 1.2 截取要预测的time字段的 月份信息
    pre_month = time[5:7]
    month_list = []
    for i in range(24,36):
        if pre_month == feature_cols[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    # 2.解析窗口特征
    last_1h_time = (pd.to_datetime(time) - pd.Timedelta("1h")).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load = data_dict.get(last_1h_time, 500)

    last_2h_time = (pd.to_datetime(time) - pd.Timedelta("2h")).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 500)

    last_3h_time = (pd.to_datetime(time) - pd.Timedelta("3h")).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 500)

    # 3.解析昨日同时刻特征
    yesterday_time = (pd.to_datetime(time) - pd.Timedelta("1D")).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load = data_dict.get(yesterday_time, 500)

    # 4.拼接特征数据
    feature_data = hour_list + month_list + [last_1h_load, last_2h_load, last_3h_load, yesterday_load]

    # 5.转成df对象
    feature_df = pd.DataFrame([feature_data], columns=feature_cols)
    return feature_df

# 3.数据可视化
def result_ana(data):
    """
    结果分析, 绘制预测时间-真实负荷折线图,  预测时间-预测负荷折线图
    :param data: 预测结果 ['预测时间', '真实负荷', '预测负荷']
    :return:
    """
    plt.figure(figsize=(30,20))
    plt.plot(data["预测时间"], data["真实负荷"], label="真实负荷")
    plt.plot(data["预测时间"], data["预测负荷"], label="预测负荷")
    plt.title("预测结果与实际结果对比分析")
    plt.xlabel("时间")
    plt.ylabel("负荷")
    plt.legend()
    # 横坐标时间若不处理太过密集，这里调大时间展示的间隔
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))
    # 时间展示时旋转45度
    plt.xticks(rotation=45)
    plt.savefig("../data/fig/预测结果与实际结果对比分析.png")
    plt.show()

# 4.测试
if __name__ == '__main__':
    # 4.1创建电力负荷预测类对象
    plp = PowerLoadPredict()
    # 4，2加载模型对象
    estimator = joblib.load("../model/xgb_20261122.pkl")
    # 4.3 确定要预测的时间段（2025-08-01 00：00：00及以后的时间）
    pre_times = plp.data["time"][plp.data["time"] >= "2015-08-01 00:00:00"]
    evaluate_list = []
    # 4.4 为了模拟实际预测场景，把预测的时间以及以后的负荷都遮盖掉，因此新建一个数据字典，只保存预测时间以前的数据字典
    for pre_time in pre_times:
        plp.logger.info(f"正在预测 {pre_time}时间的负荷")
        time_load_dict_masked= {k : v for k, v in plp.time_load_dict.items() if k < pre_time}
        # print(time_load_dict_masked)
        # 4.5预测负荷
        # 4.5.1 解析特征（定义解析特征方法）
        feature_df = feature_extract(time_load_dict_masked, pre_time, plp.logger)
        # 4.5.2 利用加载的模型预测
        y_pred = estimator.predict(feature_df)
        print(f"预测结果为：{y_pred}")
        # 4.6保存预测时间对应的真实负荷
        true_load = plp.time_load_dict.get(pre_time, 500)
        # 4.7 结果保存到evaluate_list，三个元素分别是预测时间、真实负荷、预测负荷，方便后续进行预测结果评价
        evaluate_list.append([pre_time, true_load, y_pred[0]])
        plp.logger.info(f"预测时间：{pre_time}, 真实负荷：{true_load}, 预测负荷：{y_pred[0]}")
    # 4.8 循环结束后，evaluate_list转为DataFrame
    evaluate_df = pd.DataFrame(evaluate_list, columns=["预测时间", "真实负荷", "预测负荷"])
    # 5.预测结果评价
    # 5.1 计算预测结果与真实结果的MAE
    print("平均绝对误差:", mean_absolute_error(evaluate_df["真实负荷"], evaluate_df["预测负荷"]))
    # 5.2 绘制折线图（预测时间-真实负荷折线图，预测时间-预测负荷折线图），查看预测效果
    result_ana(evaluate_df)
