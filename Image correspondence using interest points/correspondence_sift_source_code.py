import numpy as np
import cv2

img_1 = cv2.imread('1.jpg') # open image 1
img_2 = cv2.imread('2.jpg') # open image 2
gray1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

#Calculating the euclidean distance between the descriptors and setting threshold value 
l1 = []
l2 = []
for i in range(0, len(des1)):
    for j in range (0, len(des2)):
        diff = np.subtract(des1[i], des2[j])
        sqr = np.power(diff,2)
        E = np.power(np.sum(sqr),0.5)

        if (E < 50):
            l1.append([kp1[i].pt[0],kp1[i].pt[1]])
            l2.append([kp2[j].pt[0],kp2[j].pt[1]]) 
#Creating a blank image to map point correspondence
           
f_imgshape1 = img_1.shape[1]+img_2.shape[1]
img_new = np.zeros((img_1.shape[0],f_imgshape1, 3), dtype ='uint8' )

#Mapping point correspondence between the two images 
for row in range (0, img_new.shape[0]-1): 
    for col in range (0, img_new.shape[1]-1):
        if col < img_1.shape[1]:
            img_new[row][col] = img_1[row][col]
        if  col > img_1.shape[1] and col < img_new.shape[1]:
            img_new[row][col] = img_2[row][col-img_1.shape[1]]
i = 0 
for i in range (0, len(l1)):
    cv2.line(img_new,(int(l1[i][0]),int(l1[i][1])),(int(l2[i][0]+img_1.shape[1]),int(l2[i][1])),(255,0,0),1)
cv2.imwrite('SIFT.jpg' , img_new) 