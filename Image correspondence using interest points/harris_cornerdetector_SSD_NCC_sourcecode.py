import numpy as np
import cv2
#Function to sort a list based off the third element 
def takethird(elem):
    return elem[2]
#Harris corner detector function
def corner(sigma,img_1, name, grayimg_1):
    Dx = cv2.Sobel(grayimg_1,cv2.CV_64F,1,0,ksize=1)
    Dy = cv2.Sobel(grayimg_1,cv2.CV_64F,0,1,ksize=1)
    Dx_2 = np.square(Dx)
    Dy_2 = np.square(Dy)
    Dxy = Dx 
    for i in range(0,len(Dxy)):
        for j in range(0,len(Dxy[0])):
             Dxy[i][j] = Dxy[i][j]*Dy[i][j]
    N = 5*sigma
    if (N%2) == 0:
        N = N+1
    x = int(N/2)
    a = 0
    b = 0
    C = list()
    r = np.zeros(0)
    for a in range (x, img_1.shape[0] - x):
        for b in range(int(N/2), img_1.shape[1] - x):
            if (Dx[a,b] != 0) and (Dy[a,b] != 0) and (Dy[a,b]%Dx[a,b] != 0): 
                sdx_2 = Dx_2[a-x:a+x+1, b-x:b+x+1]
                sdy_2 = Dy_2[a-x:a+x+1, b-x:b+x+1]   
                sdxy = Dxy[a-x:a+x+1, b-x:b+x+1]
                sumdx_2 = np.sum(sdx_2)
                sumdy_2 = np.sum(sdy_2)
                sumdxy = np.sum(sdxy)
                trace = sumdx_2 + sumdy_2
                det = sumdx_2*sumdy_2 - sumdxy**(2)
                R = det - ((0.05) * trace**(2))
                r = np.append(r,R)
                if R > 10000:
                    C.append([a,b,R])
    C.sort(key = takethird, reverse = True)
#Removes corner points close to one another
    C2 = []
    C1 = []
    C2.append(C[0])
    nna = 1
    c = 0
    for a in range (1, 10000):
        for b in range (0,nna): 
            if ((C[a][0] < C2[b][0]+10) and (C[a][0]>C2[b][0]-10)) and ((C[a][1] < C2[b][1]+10) and (C[a][1]>C2[b][1]-10)) :  
                break
            else:
                c = c  + 1 
        if (c==nna)and(c!= 0):
            C2.append(C[a])
            nna = nna + 1
        c = 0
        if (nna == 100):
            break
    C1 = C2            
    img_2 = img_1.copy()
    for k in range(0,len(C1)):
        cv2.circle(img_2,(C1[k][1], C1[k][0]), 2, (0,0,255), thickness = 3 )
    cv2.imwrite('%s.jpg' %(name), img_2) 
    return C1
#Function to calculate SSD between corner points and map interest points 
def SSD(C1,C2,img_1, img_2, name, grayimg_1, grayimg_2):
#Finding min SSD value to set the threshold
    j = 0
    i = 0
    S = np.zeros(0)
    for i in range(0, len(C1)):
        a = C1[i][0]
        b = C1[i][1]
        win_1 = grayimg_1[a-15:a+15+1, b-15:b+15+1]
        win_1 = win_1.astype('int64')
        if (win_1.size != 961):
            continue 
        for j in range (0, len(C2)):
            c = C2[j][0]
            d = C2[j][1]
            win_2 = grayimg_2[c-15:c+15+1, d-15:d+15+1]
            win_2 = win_2.astype('int64')
            if (win_2.size != 961):
                continue
            SSD = np.sum(np.square(win_1-win_2))
            S = np.append(S,SSD)
    M = min(S)
 #Finding matching interest points
    l1 = []
    l2 = []
    ll1 = []
    ll2 = []
    j = 0
    i = 0
    for i in range(0, len(C1)):
        temp1 = []
        temp2 = []
        a = C1[i][0]
        b = C1[i][1]
        win_1 = grayimg_1[a-15:a+15+1, b-15:b+15+1]
        win_1 = win_1.astype('int64')
        if (win_1.size != 961):
            continue 
        for j in range (0, len(C2)):
            c = C2[j][0]
            d = C2[j][1]
            win_2 = grayimg_2[c-15:c+15+1, d-15:d+15+1]
            win_2 = win_2.astype('int64')
            if (win_2.size != 961):
                continue
            SSD = np.sum(np.square(win_1-win_2))
            temp1.append([a,b, SSD])
            temp2.append([c,d, SSD])
        temp1.sort(key = takethird)
        temp2.sort(key = takethird)
        ll1.append(temp1[0])
        ll2.append(temp2[0])  
    for k in range(0, len(ll1)):
        if (ll1[k][2] < 10 * M ):
            l1.append(ll1[k])
            l2.append(ll2[k])             
#Mapping point correspondence between the two images 
    f_imgshape1 = img_1.shape[1]+img_2.shape[1]

    img_new = np.zeros((img_1.shape[0],f_imgshape1, 3), dtype ='uint8' )

    for row in range (0, img_new.shape[0]-1): 
     for col in range (0, img_new.shape[1]-1):
      if col < img_1.shape[1]:
        img_new[row][col] = img_1[row][col]
      if  col > img_1.shape[1] and col < img_new.shape[1]:
        img_new[row][col] = img_2[row][col-img_1.shape[1]]
    i = 0 
    for i in range (0, len(l1)):
        cv2.line(img_new,(l1[i][1],l1[i][0]),(l2[i][1]+img_1.shape[1],l2[i][0]),(255,0,0),1)
    cv2.imwrite('SSD_%s.jpg' %(name), img_new) 
#Function to calculate NCC between corner points and map interest points 
def NCC(C1,C2,img_1, img_2, name,grayimg_1,grayimg_2):
 #Finding matching interest points     
    l1 = []
    l2 = []
    ll1 = []
    ll2 = []
    temp1 = []
    temp2 = []
    j = 0
    i = 0
    for i in range(0, len(C1)):
        temp1 = []
        temp2 = []
        a = C1[i][0]
        b = C1[i][1]
        win_1 = grayimg_1[a-15:a+15+1, b-15:b+15+1] 
        if (win_1.size != 961):
            continue 
        for j in range (0, len(C2)):
            c = C2[j][0]
            d = C2[j][1]
            win_2 = grayimg_2[c-15:c+15+1, d-15:d+15+1]
            if (win_2.size != 961):
                continue
            win_1 = win_1.astype('int64')
            win_2 = win_2.astype('int64')
            m1 = np.mean(win_1)
            m2 = np.mean(win_2)
            win1_m = np.subtract(win_1,m1)
            win2_m = np.subtract(win_2,m2)
            num = np.sum(np.multiply(win1_m,win2_m))
            den = np.sqrt(np.sum(np.square(win1_m)) * np.sum(np.square(win1_m))) 
            NCC = np.divide(num,den)
            temp1.append([a,b, NCC])
            temp2.append([c,d, NCC])
        temp1.sort(key = takethird, reverse = True)
        temp2.sort(key = takethird, reverse = True)
        ll1.append(temp1[0])
        ll2.append(temp2[0])       
#setting the thershold for NCC values
    for k in range(0, len(ll1)):
        if (ll1[k][2] > 0.99)and(ll1[k][2] < 1.02 ):
            l1.append(ll1[k])
            l2.append(ll2[k])   
#Mapping point correspondence between the two images 

    f_imgshape1 = img_1.shape[1]+img_2.shape[1]

    img_new = np.zeros((img_1.shape[0],f_imgshape1, 3), dtype ='uint8' )

    for row in range (0, img_new.shape[0]-1): 
     for col in range (0, img_new.shape[1]-1):
      if col < img_1.shape[1]:
        img_new[row][col] = img_1[row][col]
      if  col > img_1.shape[1] and col < img_new.shape[1]:
        img_new[row][col] = img_2[row][col-img_1.shape[1]]
    
    i = 0 
    for i in range (0, len(l1)):
        cv2.line(img_new,(l1[i][1],l1[i][0]),(l2[i][1]+img_1.shape[1],l2[i][0]),(255,0,0),1)
    cv2.imwrite('%s.jpg' %(name), img_new) 
    
sigma = 3

img_1 = cv2.imread('1.jpg')
img_2 = cv2.imread('2.jpg')
blur1 = cv2.GaussianBlur(img_1,(15,15),0)
grayimg_1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
blur2 = cv2.GaussianBlur(img_2,(15,15),0)
grayimg_2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

str1 = 'cpp1_3'
C1 = corner(sigma,img_1,str1,grayimg_1)    
str2 = 'cpp2_3'
C2 = corner(sigma,img_2,str2,grayimg_2)
str3 = 'ssd_3'
SSD(C1,C2,img_1, img_2, str3, grayimg_1, grayimg_2)
str4 = 'NCC_3'
NCC(C1,C2,img_1, img_2, str4, grayimg_1, grayimg_2)

