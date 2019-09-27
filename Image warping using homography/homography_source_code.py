Source Code for Task (a)
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

img_1 = cv2.imread('1.jpg', 1)
img_2 = cv2.imread('2.jpg', 1)
img_3 = cv2.imread('3.jpg', 1)
img_4 = cv2.imread('Jackie.jpg', 1)


pt_img1 = np.array ([[167,1509,1], [717,2950,1], [2238,1488,1], [2045,3000,1]])
pt_1 = np.array ([[1509,167], [2950,717], [1488,2238], [3000,2045]])
pt_img2 = np.array([[333,1323,1], [611,3020,1],[2011,1301,1],[1903,3031,1]])
pt_2 = np.array([[1323,333],[3020,611],[1301,2011],[3031,1903]])
pt_img3 = np.array([[728,918,1],[386,2791,1],[2087,895,1],[2224,2854,1]])
pt_3 = np.array([[918,728],[2791,386],[895,2087],[2854,2224]])
pt_img4 = np.array([[0,0,1],[0,1280,1],[719,0,1],[719,1280,1]])

#Building the homography matrix

def homography (pt_f, pt_i):
    T = np.zeros((8,8))
    s = np.zeros((8,1))
    for i in range (1,5):
     T[2*i-2] = [pt_i[i-1][0],pt_i[i-1][1],1, 0, 0, 0, (-1*pt_i[i-1][0]*pt_f[i-1][0]), (-1*pt_i[i-1][1]*pt_f[i-1][0])]
     T[2*i-1] = [0, 0, 0, pt_i[i-1][0], pt_i[i-1][1], 1, (-1*pt_i[i-1][0]*pt_f[i-1][1]), (-1*pt_i[i-1][1]*pt_f[i-1][1])]
     s [2*i-2] = pt_f[i-1][0]
     s [2*i-1] = pt_f[i-1][1]

    H = np.zeros((3,3))
    inv_T = np.linalg.inv(T)
    h = np.matmul(inv_T, s)
    H[0] = h[0:3,0]
    H[1][0] = h[3][0]
    H[1][1] = h[4][0]
    H[1][2] = h[5][0]
    H[2][0] = h[6][0]
    H[2][1] = h[7][0]
    H[2][2] = 1
    return H
    
#creating the bounding region
def bound(img,pt):
    temp = np.zeros((img.shape[0], img.shape[1], 3), dtype ='uint8' )
    pt_temp = np.array ([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]], [pt[3][0], pt[3][1]], [pt[2][0], pt[2][1]]], np.int32)
    temp = cv2.fillPoly(temp,[pt_temp],(255,255,255))
    return temp
    
#mapping the images
def image_loop (img_dest, img_src, he, bound):
    temp = np.ones(3)
    for row in range (0, img_dest.shape[0]-1): 
     for col in range (0, img_dest.shape[1]-1):
      if bound[row, col, 0] == 255 and bound[row, col, 1] == 255 and bound[row, col, 2] == 255:
       temp[0] = row
       temp[1] = col
       new_rc = np.matmul(he,temp)
       new_rc = np.divide(new_rc,new_rc[2])
       if (new_rc[0]>0) and (new_rc[0]<img_src.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_src.shape[1]):
        img_dest[row][col] = img_src[math.floor(new_rc[0]),math.floor(new_rc[1])]
    return img_dest
    
H_1 = homography(pt_img4, pt_img1)
b_1 = bound(img_1, pt_1)
out_1 = image_loop(img_1, img_4, H_1, b_1)
cv2.imwrite('output_1.jpg',out_1)
   
H_2 = homography(pt_img4, pt_img2)
b_2 = bound(img_2, pt_2)
out_2 = image_loop(img_2, img_4, H_2, b_2)
cv2.imwrite('output_2.jpg',out_2)

H_3 = homography(pt_img4, pt_img3)
b_3 = bound(img_3, pt_3)
out_3 = image_loop(img_3, img_4, H_3, b_3)
cv2.imwrite('output_3.jpg',out_3)

Source Code for Task (b)
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

img_1 = cv2.imread('1.jpg', 1)
img_2 = cv2.imread('2.jpg', 1)
img_3 = cv2.imread('3.jpg', 1)
img_4 = cv2.imread('Jackie.jpg', 1)


pt_img1 = np.array ([[167,1509,1], [717,2950,1], [2238,1488,1], [2045,3000,1]])

pt_img2 = np.array([[333,1323,1], [611,3020,1],[2011,1301,1],[1903,3031,1]])

pt_img3 = np.array([[728,918,1],[386,2791,1],[2087,895,1],[2224,2854,1]])


#Building the homography matrix

def homography (pt_f, pt_i):
    T = np.zeros((8,8))
    s = np.zeros((8,1))
    for i in range (1,5):
     T[2*i-2] = [pt_i[i-1][0],pt_i[i-1][1],1, 0, 0, 0, (-1*pt_i[i-1][0]*pt_f[i-1][0]), (-1*pt_i[i-1][1]*pt_f[i-1][0])]
     T[2*i-1] = [0, 0, 0, pt_i[i-1][0], pt_i[i-1][1], 1, (-1*pt_i[i-1][0]*pt_f[i-1][1]), (-1*pt_i[i-1][1]*pt_f[i-1][1])]
     s [2*i-2] = pt_f[i-1][0]
     s [2*i-1] = pt_f[i-1][1]

    H = np.zeros((3,3))
    inv_T = np.linalg.inv(T)
    h = np.matmul(inv_T, s)
    H[0] = h[0:3,0]
    H[1][0] = h[3][0]
    H[1][1] = h[4][0]
    H[1][2] = h[5][0]
    H[2][0] = h[6][0]
    H[2][1] = h[7][0]
    H[2][2] = 1
    return H

H_1 = homography(pt_img2, pt_img1)
H_2 = homography(pt_img3, pt_img2)
homography = np.matmul(H_1,H_2)
new = np.zeros((img_1.shape[0], img_1.shape[1], 3), dtype ='uint8' )
temp = np.ones(3)
for row in range (0, new.shape[0]-1): 
 for col in range (0, new.shape[1]-1):
  temp[0] = row
  temp[1] = col
  new_rc = np.matmul(np.linalg.inv(homography),temp)
  new_rc = np.divide(new_rc,new_rc[2])
  if (new_rc[0]>0) and (new_rc[0]<img_1.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_1.shape[1]):
   new[row][col] = img_1[math.floor(new_rc[0]),math.floor(new_rc[1])]
cv2.imwrite('output_task2.jpg',new)

Source Code for Task (c)
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

img_1 = cv2.imread('1_3.jpg', 1)
img_2 = cv2.imread('2_3.jpg', 1)
img_3 = cv2.imread('3_3.jpg', 1)
img_4 = cv2.imread('4_3.jpg', 1)


pt_img1 = np.array ([[1119,1196,1], [1194,1974,1], [2904,1277,1], [2542,1974,1]])
pt_1 = np.array ([[1196,1119], [1974,1194], [1277,2904], [1974,2542]])
pt_img2 = np.array([[875,539,1], [829,2404,1],[2722,831,1],[2719,2219,1]])
pt_2 = np.array([[539,875],[2404,829],[831,2722],[2219,2719]])
pt_img3 = np.array([[875,662,1],[700,2023,1],[2700,767,1],[3121,1996,1]])
pt_3 = np.array([[662,875],[2023,700],[767,2700],[1996,3121]])
pt_img4 = np.array([[0,0,1],[0,1627,1],[2107,0,1],[2107,1627,1]])

#Building the homography matrix

def homography (pt_f, pt_i):
    T = np.zeros((8,8))
    s = np.zeros((8,1))
    for i in range (1,5):
     T[2*i-2] = [pt_i[i-1][0],pt_i[i-1][1],1, 0, 0, 0, (-1*pt_i[i-1][0]*pt_f[i-1][0]), (-1*pt_i[i-1][1]*pt_f[i-1][0])]
     T[2*i-1] = [0, 0, 0, pt_i[i-1][0], pt_i[i-1][1], 1, (-1*pt_i[i-1][0]*pt_f[i-1][1]), (-1*pt_i[i-1][1]*pt_f[i-1][1])]
     s [2*i-2] = pt_f[i-1][0]
     s [2*i-1] = pt_f[i-1][1]

    H = np.zeros((3,3))
    inv_T = np.linalg.inv(T)
    h = np.matmul(inv_T, s)
    H[0] = h[0:3,0]
    H[1][0] = h[3][0]
    H[1][1] = h[4][0]
    H[1][2] = h[5][0]
    H[2][0] = h[6][0]
    H[2][1] = h[7][0]
    H[2][2] = 1
    return H
    
#creating the bounding region
def bound(img,pt):
    temp = np.zeros((img.shape[0], img.shape[1], 3), dtype ='uint8' )
    pt_temp = np.array ([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]], [pt[3][0], pt[3][1]], [pt[2][0], pt[2][1]]], np.int32)
    temp = cv2.fillPoly(temp,[pt_temp],(255,255,255))
    return temp
    
#mapping the images
def image_loop (img_dest, img_src, he, bound):
    temp = np.ones(3)
    for row in range (0, img_dest.shape[0]-1): 
     for col in range (0, img_dest.shape[1]-1):
      if bound[row, col, 0] == 255 and bound[row, col, 1] == 255 and bound[row, col, 2] == 255:
       temp[0] = row
       temp[1] = col
       new_rc = np.matmul(he,temp)
       new_rc = np.divide(new_rc,new_rc[2])
       if (new_rc[0]>0) and (new_rc[0]<img_src.shape[0])and (new_rc[1]>0) and (new_rc[1]<img_src.shape[1]):
        img_dest[row][col] = img_src[math.floor(new_rc[0]),math.floor(new_rc[1])]
    return img_dest
    
H_1 = homography(pt_img4, pt_img1)
b_1 = bound(img_1, pt_1)
out_1 = image_loop(img_1, img_4, H_1, b_1)
cv2.imwrite('output_1_task3.jpg',out_1)
   
H_2 = homography(pt_img4, pt_img2)
b_2 = bound(img_2, pt_2)
out_2 = image_loop(img_2, img_4, H_2, b_2)
cv2.imwrite('output_2_task3.jpg',out_2)

H_3 = homography(pt_img4, pt_img3)
b_3 = bound(img_3, pt_3)
out_3 = image_loop(img_3, img_4, H_3, b_3)
cv2.imwrite('output_3_task3.jpg',out_3)
