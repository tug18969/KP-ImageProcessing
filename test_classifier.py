# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 20:34:39 2017

@author: Peter
"""
import sys
import cv2
from os import listdir
from os.path import isfile, join



def main():
    print(sys.version)
    print(cv2.__version__)
    print("hello")
    
    cvpath = 'E:/Temple/CIS 4398 - Projects in Computer Science/OpenCV Tests/cascadeTest/'
    cascade = cv2.CascadeClassifier(cvpath + 'watermeloncascade10stage.xml')
    cascade2 = cv2.CascadeClassifier(cvpath + '2watermeloncascade10stage.xml')
    
    #for each file in the watermelon directory
    watpath = 'E:/Temple/CIS 4398 - Projects in Computer Science/OpenCV Tests/cascadeTest/watermelon/'
    for filename in listdir(watpath):
        if filename.endswith(".png"):
            print(filename)
            img = cv2.imread(watpath+filename)
            #rescale image
            imgsmaller = cv2.resize(img,(500,500))

			#convert to greyscale
            gray = cv2.cvtColor(imgsmaller,cv2.COLOR_BGR2GRAY)
            
            			#run classifier
            cascade.detectMultiScale(gray,1.3,5)
            objs = cascade.detectMultiScale(gray,1.3,5)
            objs2 = cascade2.detectMultiScale(gray,1.3,5)
            
            			#display all rectangles
            for(x,y,w,h) in objs:
                cv2.rectangle(imgsmaller,(x,y),(x+w,y+h),(255,255,0),2)
            
            for(x,y,w,h) in objs2:
                cv2.rectangle(imgsmaller, (x,y), (x+w,y+h),(0,255,255),2)

			#show image and wait for input
            cv2.imshow("Your Image",imgsmaller)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    
if(__name__=="__main__"):main()