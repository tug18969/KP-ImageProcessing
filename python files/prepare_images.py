# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 11:45:37 2017

@author: Peter
"""
import urllib2
import cv2
import numpy as np
import os



def main():
    store_raw_images()
    create_pos_n_neg()
    
    
    
def store_raw_images():
	#this is the image_net link that contains all of the urls of negative images
    neg_images_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04081281'
    
	#open the web page
    neg_image_urls = urllib2.urlopen(neg_images_link).read().decode()
	#counter for image names
    pic_num = 758
    
	#create a neg folder
    if not os.path.exists('neg'):
        os.makedirs('neg')
        
    for i in neg_image_urls.split('\n'):
        try:
            print(i)
			#get each image
            urllib2.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
            
			#read each image in as a greyscale
            img = cv2.imread("neg/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
            
			#resize the negative images
			# should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))

			#save to directory
            cv2.imwrite("neg/"+str(pic_num)+".jpg",resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  


def create_pos_n_neg():
	#create the info and bg files
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
                    
                    
if(__name__=="__main__"):main()
