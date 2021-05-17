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



class FeatureDecomposer:
    
    def __init__(self):
        pass
    
    def decompose_features(self,config,input_df):
        
        config = config["decomposer"]
        decomposer = config["type"]
        
        if config["engine"] == "training":
            if decomposer == "PCA":            
                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(input_df)
                principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
        else:
            ## Here the saved pca model will be loaded for inerence
            pass
            
        return principalDf
        
# dc = FeatureDecomposer()
# decomposed_features = dc.decompose_features(config,scaled_df)