# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def main():
    from TestClassifier import FoodClassifier
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    direc = 'demo/'
    files = os.listdir('demo')
    model = FoodClassifier('cnnModelDEp80.h5',['apple','banana'])
    
    for file in files:
        img = Image.open(direc + file)
        plt.figure()
        plt.imshow(img)
        plt.show()
        
        print(model.predict(img))
        input('Press Enter to Continue')
        

if(__name__ == "__main__"):main()