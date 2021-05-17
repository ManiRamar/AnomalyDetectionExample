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


class SensorPreprocessor:
    
    def __init__(self,):
        pass
    
    def _calc_percent_NAs(self,df):
        
        nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent']) 
        idx = nans['percent'] > 0
        return nans[idx]

    def drop_sensors(self,df):
    
        missing_perc_df = self._calc_percent_NAs(df).reset_index()
        missing_perc_df.columns = ["sensor","percent"]
        missing_perc_df = missing_perc_df[missing_perc_df["percent"]>0.80]
        
        sensors_to_be_removed = list(missing_perc_df["sensor"].unique()) 
        
        df2 = df.copy()
        for sensor in sensors_to_be_removed:
            
            del df2[sensor]
            
        df2 = df2.dropna()
        
        return df2
          
    def preprocess_data(self,config, input_df):
        
        config = config["data_preprocessor"]
        
        df = input_df 
        df['date'] = pd.to_datetime(df[config["time_col"]])
        df.drop(columns = [config["time_col"]])
        
        if config["engine"] == "training":
            preprocessed_df = self.drop_sensors(df)
            del preprocessed_df["timestamp"]
            
        else:
            ## Placeholder for inference logic 
            pass
        
        preprocessed_df.to_csv(config["output_path"])
        
        return preprocessed_df
        
        
# preprocessed_data = sp.preprocess_data(config,loaded_data)       
# preprocessed_data