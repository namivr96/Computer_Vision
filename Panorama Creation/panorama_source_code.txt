
import numpy as np
import cv2
from scipy import optimize

#Function to sort point correspondence based on Euclidean distances
def takethird(elem):
    return elem[2]

#Function that calculates the correspondences between two images using SIFT image mapping and print an image showing the correspondences 
def SIFT(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
#calculates the correspondences
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    i = 0
    j = 0 
    l1 = []
    l2 = []
    ll1 = []
    ll2 = []
#Calculating feature mapping using euclidean distances
    for i in range(0, len(des1)):
        dist_min = np.Inf
        for j in range (0, len(des2)):
            diff = np.subtract(des1[i], des2[j])
            sqr = np.power(diff,2)
            E = np.power(np.sum(sqr),0.5)
            if E < dist_min:
                dist_min = E
                pair1 = [kp1[i].pt[0],kp1[i].pt[1], E]
                pair2 = [kp2[j].pt[0],kp2[j].pt[1], E]
        ll1.append(pair1)
        ll2.append(pair2)
    ll1.sort(key = takethird)
    ll2.sort(key = takethird)
    l1 = ll1 [0:200]
    l2 = ll2 [0:200]
#Prints an image showing the best 200 interest points between two images
    f_imgshape1 = img1.shape[1]+img2.shape[1]
    img_new = np.zeros((img1.shape[0],f_imgshape1, 3), dtype ='uint8' )
    for row in range (0, img_new.shape[0]-1): 
        for col in range (0, img_new.shape[1]-1):
            if col < img1.shape[1]:
                img_new[row][col] = img1[row][col]
            if  col > img1.shape[1] and col < img_new.shape[1]:
                img_new[row][col] = img2[row][col-img1.shape[1]]
    i = 0 
    for i in range (0, len(l1)):
        cv2.line(img_new,(int(l1[i][0]),int(l1[i][1])),(int(l2[i][0]+img1.shape[1]),int(l2[i][1])),(255,0,0),1)
    cv2.imwrite('45_SIFT.jpg', img_new)
    return l1,l2

#Function that calculates the error as well as Jacobian matrix for the LM algorithm 
def func(h):
    error = 0 
    j = 0 
    f_x = []
    f_y = []
    xrw = []
    yrw = []
    error = []
    Jac = np.zeros((2*len(il1),len(h)))
#Calculates the mapped points as well as the jacobian matrix of the cost function    
    for j in range(0, len(il1)):
        x = il1[j][0]
        y = il1[j][1]
        xrw.append(il2[j][0])
        yrw.append(il2[j][1])
        
        num_x = h[0]*x + h[1]*y + h[2]
        den_x = h[6]*x + h[7]*y + h[8]
        x_new = np.divide(num_x,den_x)
        
        num_y = h[3]*x + h[4]*y + h[5]
        den_y = h[6]*x + h[7]*y + h[8]
        y_new = np.divide(num_y,den_y)
        
        f_x.append(x_new)
        f_y.append(y_new)
        
        Jac [j][0]= -x/den_x
        Jac [j][1]= -y/den_x
        Jac [j][2]= -1/den_x
        Jac [j][3]= 0
        Jac [j][4]= 0
        Jac [j][5]= 0
        Jac [j][6]= x*(num_x)*np.power(den_x,-2)
        Jac [j][7]= y*(num_x)*np.power(den_x,-2)
        Jac [j][8]= num_x*np.power(den_x,-2)
        
        Jac [j+1][0]= 0
        Jac [j+1][1]= 0
        Jac [j+1][2]= 0
        Jac [j+1][3]= -x/den_y
        Jac [j+1][4]= -y/den_y
        Jac [j+1][5]= -1/den_y
        Jac [j+1][6]= x*(num_y)*np.power(den_y,-2)
        Jac [j+1][7]= y*(num_y)*np.power(den_y,-2)
        Jac [j+1][8]= num_y*np.power(den_y,-2)
#Calculates the error        
    for i in range(0, len(f_x)): 
        errx = (xrw[i]-f_x[i])
        erry = (yrw[i]-f_y[i])
        error.append(errx)
        error.append(erry)
    return error, Jac     

#Function using openCV for the LM algorithm         
def optim_H(H):
    h = np.array([H[0][0],H[0][1],H[0][2],H[1][0],H[1][1],H[1][2],H[2][0],H[2][1],H[2][2]])
    sol = optimize.root(func, h, jac = 'True', method='lm')
    H[0][0] = sol.x[0]
    H[0][1] = sol.x[1]
    H[0][2] = sol.x[2]
    H[1][0] = sol.x[3]
    H[1][1] = sol.x[4]
    H[1][2] = sol.x[5]
    H[2][0] = sol.x[6]
    H[2][1] = sol.x[7]
    H[2][2] = sol.x[8]
    return H

#Function calculting homography between 2 images given 6 correspondence points 
def homography (pt_f, pt_i):
    T = np.zeros((12,8))
    s = np.zeros((12,1))
    for i in range (1,7):
     T[2*i-2] = [pt_i[i-1][0],pt_i[i-1][1],1, 0, 0, 0, (-1*pt_i[i-1][0]*pt_f[i-1][0]), (-1*pt_i[i-1][1]*pt_f[i-1][0])]
     T[2*i-1] = [0, 0, 0, pt_i[i-1][0], pt_i[i-1][1], 1, (-1*pt_i[i-1][0]*pt_f[i-1][1]), (-1*pt_i[i-1][1]*pt_f[i-1][1])]
     s [2*i-2] = pt_f[i-1][0]
     s [2*i-1] = pt_f[i-1][1]

    H = np.zeros((3,3))
    inv_T =np.matmul(np.linalg.inv(np.matmul(np.transpose(T),T)),np.transpose(T))
    h = np.matmul(inv_T, s)
    H[0] = h[0:3,0]
    H[1][0] = h[3][0]
    H[1][1] = h[4][0]
    H[1][2] = h[5][0]
    H[2][0] = h[6][0]
    H[2][1] = h[7][0]
    H[2][2] = 1
    return H

#Function that uses RANSAC to estimate the best homography between two images 
def RANSAC(l1,l2,N,del_t,img1,img2):
    H = []
    cl = []
    
    for tri in range(0, 80):
        indx = np.random.randint(0, len(l1)-1, 6)
        pt_i = []
        pt_f = []
        for j in indx:
            pt_i.append (l1[j])
            pt_f.append (l2[j])
        H1 = homography(pt_f,pt_i)
        counter = 0
        temp = np.ones(3)
        for j in range(0, len(l1)):
            temp[0] = l1[j][0]
            temp[1] = l1[j][1]
            new_rc = np.matmul(H1,temp)
            new_rc = np.divide(new_rc,new_rc[2])
            delta = np.sqrt(np.square(new_rc[0]-l2[j][0]) + np.square(new_rc[1]-l2[j][1]))
            if (delta < del_t):
                counter = counter + 1
        if (counter > 180):
            cl.append(counter)
            H.append(H1)
        if (len(cl) == N):
            break
    in1 = 0 
    max1 = cl[0]
    
#Determines homography that gives most number of inliers
    for a in range (0, len(cl)-1):
        if max1 < cl[a+1]:
            max1 = cl[a+1]
            in1 = a + 1
    inline1 = []
    inline2 = []
    for j in range(0, len(l1)):
            temp[0] = l1[j][0]
            temp[1] = l1[j][1]
            new_rc = np.matmul(H[in1],temp)
            new_rc = np.divide(new_rc,new_rc[2])
            delta = np.sqrt(np.square(new_rc[0]-l2[j][0]) + np.square(new_rc[1]-l2[j][1]))
            if (delta < del_t): 
                inline1.append([l1[j][0],l1[j][1]])
                inline2.append([l2[j][0],l2[j][1]])
#Printing the inliers and outliers between the two images
    f_imgshape1 = img1.shape[1]+img2.shape[1]
    img_new = np.zeros((img1.shape[0],f_imgshape1, 3), dtype ='uint8' )
    for row in range (0, img_new.shape[0]-1): 
        for col in range (0, img_new.shape[1]-1):
            if col < img1.shape[1]:
                img_new[row][col] = img1[row][col]
            if  col > img1.shape[1] and col < img_new.shape[1]:
                img_new[row][col] = img2[row][col-img1.shape[1]]
    i = 0 
    for i in range (0, len(l1)):
        cv2.line(img_new,(int(l1[i][0]),int(l1[i][1])),(int(l2[i][0]+img1.shape[1]),int(l2[i][1])),(0,0,255),1)
    i = 0
    for i in range (0, len(inline1)):
        cv2.line(img_new,(int(inline1[i][0]),int(inline1[i][1])),(int(inline2[i][0]+img1.shape[1]),int(inline2[i][1])),(0,255,0),1)
    cv2.imwrite('in_out_45.jpg' , img_new)
    j = 0
    inline1 = []
    inline2 = []

    for j in range(0, len(l1)):
            temp[0] = l1[j][0]
            temp[1] = l1[j][1]
            new_rc = np.matmul(H[in1],temp)
            new_rc = np.divide(new_rc,new_rc[2])
            delta = np.sqrt(np.square(new_rc[0]-l2[j][0]) + np.square(new_rc[1]-l2[j][1]))
            if (delta < del_t): 
                inline1.append([l1[j][0],l1[j][1]])
                inline2.append([l2[j][0],l2[j][1]])

    return H[in1],inline1,inline2
#Function to create stitching between two images        
def createimg(img_new,img_src,H):
    #H from img new to img src
    row = 0
    col = 0
    for row in range (0, img_new.shape[0]-1): 
        for col in range (0, img_new.shape[1]-1):
            temp = np.ones(3)
            temp[0] = col
            temp[1] = row 
            new_rc = np.matmul(np.linalg.inv(H),temp)
            new_rc = np.divide(new_rc,new_rc[2])
            if (new_rc[0]>0) and (new_rc[0]<img_src.shape[1])and (new_rc[1]>0) and (new_rc[1]<img_src.shape[0]):
                img_new[row][col] = img_src[int(new_rc[1]),int(new_rc[0])]
    return img_new
            
img_1 = cv2.imread('1_1.jpg') 
img_2 = cv2.imread('2_1.jpg') 
img_3 = cv2.imread('3_1.jpg') 
img_4 = cv2.imread('4_1.jpg')
img_5 = cv2.imread('5_1.jpg')
#Function call to create homgraphy between adjacent images
N = 6
del_t = 20
H_tran = np.array([[1,0,1200],[0,1,0],[0,0,1]])

l1,l2 = SIFT(img_1,img_2) 
H1_2,il1,il2 = RANSAC(l1,l2,N,del_t,img_1,img_2) 
H1_2 = optim_H(H1_2)

l1,l2 = SIFT(img_2,img_3) 
H2_3,il1,il2 = RANSAC(l1,l2,N,del_t,img_2,img_3)
H2_3 = optim_H(H2_3)

l1,l2 = SIFT(img_3,img_4) 
H3_4,il1,il2 = RANSAC(l1,l2,N,del_t,img_3,img_4)
H3_4 = optim_H(H3_4)

l1,l2 = SIFT(img_4,img_5) 
H4_5,il1,il2 = RANSAC(l1,l2,N,del_t,img_4,img_5) 
H4_5 = optim_H(H4_5)
#Creating homography between the four images and the third image
H1_3 = np.matmul(H_tran,np.matmul(H2_3,H1_2))
H2_3 = np.matmul(H_tran,H2_3)
H4_3 = np.matmul(H_tran,np.linalg.inv(H3_4))
H5_3 = np.matmul(H_tran,np.matmul(np.linalg.inv(H3_4),np.linalg.inv(H4_5)))
H3 = np.matmul(H_tran,np.array([[1,0,0],[0,1,0],[0,0,1]]))
#Creates a blank image
new = np.zeros((2*img_1.shape[0],5*img_1.shape[1],3), dtype ='uint8' )
#Creates a panaroma
img_n1 = createimg(new,img_1,H1_3)
print('img1')
img_n2 = createimg(img_n1,img_2,H2_3)
print('img2')
img_n3 = createimg(img_n2,img_3,H3)
print('img3')
img_n4 = createimg(img_n3,img_4,H4_3)
print('img4')
img_n5 = createimg(img_n4,img_5,H5_3)

cv2.imwrite('panaroma.jpg',img_n5)

