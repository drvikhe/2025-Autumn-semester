import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
import numpy as np
import platform

#字体设置
system_name = platform.system()
if system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
elif system_name == 'Windows':  # Windows
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']

plt.rcParams['axes.unicode_minus'] = False



class WeatherDataAnalyzer:
    def __init__(self, city_name="Beijing"):
        self.city_name = city_name
        self.data = None

    def fetch_data_mock(self, days=365):
        """
        模拟爬取过去一年的天气数据。
        """
        print(f"正在爬取 {self.city_name} 过去 {days} 天的天气数据...")

        dates = [datetime.today() - timedelta(days=x) for x in range(days)]
        data_list = []

        for date in dates:
            month = date.month
            base_temp = 25 - abs(month - 7) * 5
            high = base_temp + random.randint(0, 8)
            low = base_temp - random.randint(3, 10)
            weather_type = random.choice(['晴', '多云', '阴', '小雨', '中雨', '大雨'])

            # 制造脏数据
            if random.random() < 0.02:
                high = 999
            if random.random() < 0.02:
                low = None

            data_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'high_temp': high,
                'low_temp': low,
                'weather': weather_type
            })

        self.data = pd.DataFrame(data_list)
        print("数据抓取完成，共获取 {} 条记录。".format(len(self.data)))

    def clean_data(self):
        """
        数据清洗：处理缺失值、异常值和类型转换
        """
        if self.data is None:
            return

        print("\n开始数据清洗...")
        df = self.data.copy()

        missing_count = df['low_temp'].isnull().sum()
        df['low_temp'] = df['low_temp'].ffill()
        print(f"- 填充了 {missing_count} 个缺失的低温数据。")

        df['high_temp'] = df['high_temp'].astype(float)

        abnormal_mask = df['high_temp'] > 60
        abnormal_count = abnormal_mask.sum()
        mean_high = df[df['high_temp'] <= 60]['high_temp'].mean()

        df.loc[abnormal_mask, 'high_temp'] = mean_high
        print(f"- 修正了 {abnormal_count} 个异常的高温数据。")

        # 计算温差
        df['temp_diff'] = df['high_temp'] - df['low_temp']

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])

        self.data = df
        print("数据清洗完成。")

    def visualize_data(self):
        """
        数据可视化
        """
        if self.data is None:
            return

        print("\n正在生成可视化图表...")
        df = self.data.sort_values('date')

        plt.figure(figsize=(15, 6))

        # 子图1：全年温度变化趋势
        plt.subplot(1, 2, 1)
        plt.plot(df['date'], df['high_temp'], label='最高温', color='red', alpha=0.6)
        plt.plot(df['date'], df['low_temp'], label='最低温', color='blue', alpha=0.6)
        plt.title(f'{self.city_name} 全年温度变化趋势')
        plt.xlabel('日期')
        plt.ylabel('温度 (℃)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # 子图2：天气类型分布饼图
        plt.subplot(1, 2, 2)
        weather_counts = df['weather'].value_counts()
        plt.pie(weather_counts, labels=weather_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('天气类型分布')

        plt.tight_layout()
        save_name = f'{self.city_name}_weather_analysis.png'
        plt.savefig(save_name)
        print(f"图表已保存为 {save_name}")

    def run(self):
        self.fetch_data_mock()
        self.clean_data()
        self.visualize_data()
        return self.data.head()


if __name__ == "__main__":
    analyzer = WeatherDataAnalyzer(city_name="Shanghai")
    head_data = analyzer.run()
    print("\n处理后的前5条数据预览：")
    print(head_data)