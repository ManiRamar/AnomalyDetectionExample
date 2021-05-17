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



class SensorLoader:
    
    def __init__(self,):
        pass

    def load_data(self,config):
        
        config = config["data_loader"]
        if config["engine"] == "training":
            df = pd.read_csv(config["data_path"],index_col=[0])
            df = df.drop_duplicates()
        else:
            ## Placeholder for inference logic 
            pass
        
        return df

# sl = SensorLoader()
# loaded_data = sl.load_data(config)