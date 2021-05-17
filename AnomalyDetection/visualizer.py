import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime


class DataVisualizer:
    
    def __init__(self,):
        pass
        
    def plot_sensor_anomalies(self, data , sensor_name , anomaly_label):
        
        a = data[data['anomaly_label'] == anomaly_label] #anomaly
        _ = plt.figure(figsize=(18,6))
        _ = plt.plot(data[sensor_name], color='blue', label='Normal')
        _ = plt.plot(a[sensor_name], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
        _ = plt.xlabel('Date and Time')
        _ = plt.ylabel('Sensor Reading')
        _ = plt.title(sensor_name + 'Anomalies')
        _ = plt.legend(loc='best')
        plt.show();