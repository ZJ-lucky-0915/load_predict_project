# 该工具类的目的是：对数据进行预处理 ——> 时间格式化，按时间升序排序
# 数据集在 data目录下的 train.csv文件中 ——> 拆分 训练集 测试集
# 测试集在 data目录下的test.csv文件中 ——> 模拟项目上线和真实的测试集

import pandas as pd


# 定义函数 data_preprocessing() 对数据进行预处理
def data_preprocessing(filemath):
    # 1. 加载数据集
    data = pd.read_csv(filemath)

    # 2. 时间格式化
    data["time"] = pd.to_datetime(data["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 3. 按时间升序排序
    data = data.sort_values(by="time", ascending=True)

    # 4. 去重
    data = data.drop_duplicates()

    return data
if __name__ == "__main__":
    data_preprocessing()
