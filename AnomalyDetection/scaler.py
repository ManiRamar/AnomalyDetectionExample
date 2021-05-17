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


class Scaler:
    
    def __init__(self,):
        pass
    
    def scale_features(self,config,input_df):
        
        config = config["scaler"]
        input_df = input_df.set_index(config["time_col"])
        input_df = input_df.drop(columns = [config["drop_cols"]])
        scaler = config["type"]
        
        if config["engine"] == "training":
            if scaler == "StandardScaler":
                sc = StandardScaler()
        
            scaled_df = input_df.copy() 
            sc.fit(scaled_df)
            
        else:
            ### Incase of inference the saved model will be loaded and the scaled df will be generated
            pass
        
        return scaled_df
    
# sc = Scaler()
# scaled_df=sc.scale_features(config,preprocessed_data)