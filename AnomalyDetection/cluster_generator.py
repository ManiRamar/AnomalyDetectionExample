import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


class SensorStateClusterGenerator:
    
    def __init__(self,):
        pass
    
    # Write a function that calculates distance between each point and the centroid of the closest cluster
    def _getDistanceByPoint(self,data, model):
        """ Function that calculates the distance between a point and centroid of a cluster, 
                returns the distances in pandas series"""
        distance = []
        for i in range(0,len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[model.labels_[i]-1]
            distance.append(np.linalg.norm(Xa-Xb))
        return pd.Series(distance, index=data.index)
    
    def get_anomaly_label(self,config , model):
        
        outliers_fraction = 0.13
        distance = self._getDistanceByPoint(self.input_df, model)
        number_of_outliers = int(outliers_fraction*len(distance))
        threshold = distance.nlargest(number_of_outliers).min()
        
        # anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly) 
        self.input_df['anomaly_label'] = (distance >= threshold).astype(int)
            
    def kmeans(self,config,input_df):
        
        kmeans = KMeans(n_clusters=config["sensor_state_cluster"]["n_clusters"], random_state=42)
        kmeans.fit(input_df.values)
        labels = kmeans.predict(input_df.values)
        unique_elements, counts_elements = np.unique(labels, return_counts=True)
        clusters = np.asarray((unique_elements, counts_elements))
        
        return kmeans , clusters
        
    
    def generate_cluster(self,config,input_df):
        
        cluster = config["sensor_state_cluster"]["type"]
        self.input_df = input_df
        
        if cluster == "KMeans":
            model , clusters = self.kmeans(config,input_df)
            
        self.get_anomaly_label(config , model)
        base_data = pd.read_csv(config["sensor_state_cluster"]["base_data_path"])
        base_data["anomaly_label"] = self.input_df["anomaly_label"]
        
        feats = []
        for i in list(base_data.columns):
            if "sensor" in i:
                feats.append(i)
        
        
        if config["sensor_state_cluster"]["plot_anomalies"] == "True":
            
            viz = DataVisualizer()
            for sensor in feats:
                viz.plot_sensor_anomalies(base_data , sensor , 1)           
        
        return base_data

# scg = SensorStateClusterGenerator()
# labels = scg.generate_cluster(config,decomposed_features)