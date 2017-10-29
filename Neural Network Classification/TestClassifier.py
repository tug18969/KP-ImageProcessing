# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import cv2
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np    
import os

class FoodClassifier:
#Class Attributes:
#model - the underlying keras model
#labels - the labels to be associated with the activation of each output neuron. 
#Labels must be the same size as the output layer of the neural network.    

    def __init__(self, modelpath, labels):
        self.model = load_model(modelpath)
        self.labels = labels
    
    def predict(self,img):
        
        #check if image is a filepath
        if(isinstance(img,str)):
            if(not os.path.exists(img)):
                print("Error: Invalid File Path")
                return ""
            else:
                #if its a filepath, convert to PIL image
                img = Image.open(img)
        
        #resize image
        imgr = img.resize((128,128))
        x = img_to_array(imgr).reshape((1,128,128,3))
        
        #predict
        prediction = self.model.predict(x)
        #print(prediction.shape)
        #print(prediction)
        
        #get max of predictions and return label(s)
        predIdx = np.argmax(prediction[0,:])
        return self.labels[predIdx]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        