# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 10:37:10 2020

@author: Andrew
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import pydicom
from os import listdir
from os.path import isfile, join
import pandas as pd

#Class which will get all the data required into a pandas dataframe
class DataProcessingUnit():
    
    def __init__(self, fpath,fname):
        self.fpath = fpath
        self.fileList = [f for f in listdir(fpath) if isfile(join(fpath, f))] #gets list of all files in folder
        self.fname = fname
        self.df = None
    
    def generateDF(self):
        df = pd.DataFrame(pd.read_csv(self.fname))
        df = df.head(100)
        bigList = []
        for i in range(0,54,6):
            imageid = df['ID'].iloc[i].split('_')[1]
            array = df['Label'].iloc[[i,i+1,i+2,i+3,i+4,i+5]].to_list()
            bigList.append([imageid,array])
        bigDF = pd.DataFrame(bigList,columns=['ID','probs'])
        self.df = bigDF
        self.getPixelData()
        return
    
    def getPixelData(self):
        pixelData = []
        for i in range(len(self.fileList)):
            ds = pydicom.dcmread('hemo_pics/'+self.fileList[i])
            pixel_array_numpy = ds.pixel_array
            pixelData.append(pixel_array_numpy)
        self.df['ImagePixels'] = pixelData
        return
    



        
        
            
            
    
            