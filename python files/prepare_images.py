# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 11:45:37 2017

@author: Peter
"""
import urllib
import cv2
import numpy as np
import os
import subprocess
import math
import mergevec


def main():
    link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07739125"
    direc = "apple"
    numNeg = 2549
    #store_raw_images(link,direc)
#    create_pos_n_neg()
    #found = find_uglies(direc,"ugly.jpg")
    #print "Ugly images removed:",found
    #create_bg('neg')
    info = "info"
    bg = "bg.txt"
    num_samples = 50
    print "Creating Vector File..."
    vecpath = create_positives_new(direc,info, bg, num_samples)
    print "Vector File Created."
    
    print "Training..."
    train( vecpath, bg, num_samples, numNeg, 10,"data")
    print "Training Complete."
    
def store_raw_images(images_link, store_path, pic_num=1, img_dim=(100,100), quiet=False):
    #Retrieves all images from the provided link, storing them as a jpg with an
    #incremental number in the specified folder.
    #Arguments: 
        #images_link: The url to the image-net page containing all image links
        #store_path: the directory to store all images.
        #pic_num: Counter for image names. Starts from 0 by default
        #img_dim: dimension to store the image as. All images will be greyscale.
        #quiet: whether or not to hide output.
    #Returns: returns the counter for pictures
    
    #open the web page
    neg_image_urls = urllib.urlopen(images_link).read().decode()
    
	 #create the directory if it doesnt exist
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    
    for i in neg_image_urls.split('\n'):
        try:
            
            #print img name if requested
            if(quiet==False):
                print(i)
			  
            #image reference
            img_path = store_path + '/' + str(pic_num) + ".jpg"
            
            #get each image
            urllib.urlretrieve(i, img_path)
            
            #read each image in as a greyscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            #resize the images
            resized_image = cv2.resize(img, img_dim)

            #save to directory
            cv2.imwrite(img_path, resized_image)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  
            
    #return number of pictures up to this point
    return pic_num


def create_bg(neg, bg='bg.txt'):
	#create the bg file that lists all negative images
    #Arguments:
        #neg: the negative images folder
        #bg: the txt file to contain all negative image paths
    for img in os.listdir(neg):
        line = neg + '/' + img + '\n'
        with open(bg,'a') as f:
            f.write(line)
                    
                    
def find_uglies(folder, ugly):
    #Removes images similar to the specified ugly image
    #Arguments:
        #folder: The folder to remove images from
        #ugly: The image to remove duplicates of
    #Returns: The number of ugly images removed
    
    #get the ugly image
    try:
        ug = cv2.imread(ugly)
    except Exception as e:
        print(str(e))
        return -1
    
    #move through the folder
    num_uglies = 0
    for img in os.listdir(folder):
        try:
            #load image into memory
            current_img_path = str(folder) + '/' + str(img)
            question = cv2.imread(current_img_path)
            
            #if match
            if ug.shape == question.shape and not(np.bitwise_xor(ug,question).any()):
                num_uglies +=1
                os.remove(current_img_path)
        except Exception as e:
            print(str(e))
            
    return num_uglies


def create_positives(pos,info, bg, num_samples):
    #get the number of samples that should be generated from each positive
    num_positives = len(os.listdir(pos))
    if (num_samples <= num_positives):
        samples_per_positive = 1
        samples_remainder = 0
    else:
        samples_per_positive = math.floor(num_samples/len(os.listdir(pos)))
        samples_remainder = num_samples%len(os.listdir(pos))
    
    isFirst = True
    count = 1
    for img in os.listdir(pos):
        if(count > num_samples+1):
            break
        if isFirst:
            
            exc = ["opencv_createsamples", "-img",pos + '/' + img, "-bg", bg, "-info",
                   info+"/info" + str(count) + ".lst", "-pngoutput",info,
                   "maxxangle", "0.5", "-maxyangle", "0.5", "-maxzangle", "0.5",
                   "-num", str(samples_per_positive+samples_remainder)]
            isFirst = False
            count += samples_per_positive + samples_remainder
            subprocess.call(exc)
        else:
            exc = ["opencv_createsamples", "-img",pos + '/' + img, "-bg", bg, "-info",
                   info+"/info" + str(count) + ".lst", "-pngoutput",info,
                   "maxxangle", "0.5", "-maxyangle", "0.5", "-maxzangle", "0.5",
                   "-num", str(samples_per_positive)]
            count += samples_per_positive
            subprocess.call(exc)            



def create_positives_new(pos,info,bg,num_samples):
   
    vecpath = info + "/vec"
    if not os.path.exists(vecpath):
        os.makedirs(vecpath)
        
    positives = os.listdir(pos)
    #case: fewer requested samples than positive images. Then generate
    #one sample from each positive up to the desired amount
    if(num_samples<=len(positives)):
        for i in range(0,num_samples):
            samples_path = info + '/' + str(i)
            if not os.path.exists(samples_path):
                os.makedirs(samples_path)
            
            #run create_samples
            exc = ["opencv_createsamples", "-img",pos + '/' + positives[i], "-bg", bg, "-info",
            samples_path + "/info.lst", "-pngoutput",samples_path,
            "maxxangle", "0.5", "-maxyangle", "0.5", "-maxzangle", "0.5",
            "-num", "1"]
            subprocess.call(exc)
            
            #run create_samples to make the vec file
            vecexc = ["opencv_createsamples", "-info",samples_path + "/info.lst",
                      "-num", "1","-w", "20", "-h", "20", "-vec",
                      vecpath + "/positives" + str(i) + ".vec"]
            subprocess.call(vecexc)
    
    #case: more samples requested than have positive images.
    #Generate a number of samples from each positive.
    else:
        #get minimum number of samples per positive and extra
        samples_per_positive = num_samples/len(positives)
        num_extra = num_samples%len(positives)
        
        for img in positives:
            #make a directory if needed
            samples_path = info + '/' + img.split('.')[0]
            if not os.path.exists(samples_path):
                os.makedirs(samples_path)
            
            #if we've got extras, use one
            if(num_extra!=0):
                n_samples = str(samples_per_positive+1)
                num_extra -= 1
            else:
                n_samples = str(samples_per_positive)
            
            #execute opencv_createsamples for the image
            exc = ["opencv_createsamples", "-img",pos + '/' + img, "-bg", bg, "-info",
                       samples_path + "/info.lst", "-pngoutput",samples_path,
                       "maxxangle", "0.5", "-maxyangle", "0.5", "-maxzangle", "0.5",
                       "-num", n_samples]
            subprocess.call(exc)
            
            #execute create_samples for vec file
            vecexc = ["opencv_createsamples", "-info",samples_path + "/info.lst",
                      "-num", n_samples,"-w", "20", "-h", "20", "-vec",
                      vecpath + "/positives" + img.split('.')[0] + ".vec"]
            subprocess.call(vecexc)
    
    #merge all vector files
    mergevec.merge_vec_files(vecpath, info+"/positives.vec")
    #return path to merged vector file
    return info + "/positives.vec"
                

def train(vecpath, bg, numPos, numNeg, numStages,data = "data"):
    if not os.path.exists(data):
        os.makedirs(data)
    
    exc = ["opencv_traincascade", "-data", data, "-vec", vecpath, "-bg", bg, 
           "-numPos", str(numPos), "-numNeg", str(numNeg), "-numStages", str(numStages), "-w","20",
           "-h","20"]
    errlog = open("err.txt","w")
        
    subprocess.call(exc,stdout = errlog, stderr = errlog)
    
if(__name__=="__main__"):main()
    




