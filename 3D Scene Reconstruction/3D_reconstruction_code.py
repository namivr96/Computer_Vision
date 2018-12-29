import numpy as np
import cv2
import scipy
from scipy import optimize
from scipy import linalg
from scipy.linalg import null_space
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def takethird(elem):
    return elem[2]
#Function to plot initial manual correspondence points on input images
def inlier_img(img1, img2, m_cor1, m_cor2):
    f_imgshape1 = img1.shape[1]+img2.shape[1]
    img_new = np.zeros((img1.shape[0],f_imgshape1, 3), dtype ='uint8' )
    for row in range (0, img_new.shape[0]-1): 
        for col in range (0, img_new.shape[1]-1):
            if col < img1.shape[1]:
                img_new[row][col] = img1[row][col]
            if  col > img1.shape[1] and col < img_new.shape[1]:
                img_new[row][col] = img2[row][col-img1.shape[1]]
    
    for i in range (0, len(m_cor1)):
        cv2.line(img_new,(int(m_cor1[i][0]),int(m_cor1[i][1])),(int(m_cor2[i][0]+img1.shape[1]),int(m_cor2[i][1])),(0,255,0),1)
    cv2.imwrite('inlier_rect.jpg' , img_new)
#Fuction to normalize points    
def normalize(pt):  
    mean_x = np.mean([pt[i][0] for i in range (len(pt))])
    mean_y = np.mean([pt[i][1] for i in range (len(pt))])
    dist_1 = [[pt[i][0]- mean_x,pt[i][1]- mean_y] for i in range (len(pt))]
    dist_2 = np.square(dist_1)
    dist = [np.sqrt(dist_2[i][0] + dist_2[i][1]) for i in range (len(pt))]
    mean_dist = np.sum(dist)/len(pt)
    scale = np.sqrt(2)/mean_dist
    x=-scale*mean_x;
    y=-scale*mean_y;
    T=[[scale, 0, x],[0, scale, y], [0, 0, 1]]
    pt_list = []
    for i in range(len(pt)):
        pt_temp = [pt[i][0],pt[i][1],1]
        pt_new = np.dot(T, np.transpose(pt_temp))
        pt_new = pt_new/pt_new[2]
        pt_list.append(pt_new)
    return np.array(pt_list), T
#Function to calculate fundamental matrix
def computeF(pt1,pt2,T1,T2):
    N = len(pt1)
    A = np.zeros((N,9))
    for i in range (N):
        A[i] = [pt2[i][0]*pt1[i][0], pt2[i][0]*pt1[i][1], pt2[i][0], pt2[i][1]*pt1[i][0], pt2[i][1]*pt1[i][1], pt2[i][1], pt1[i][0], pt1[i][1], 1]
    u,d,vt = np.linalg.svd(A)
    v = np.transpose(vt)
    f = v[:,-1]
    F_cond = np.reshape(f,(3,3))
    U,D,Vt = np.linalg.svd(F_cond)
    D[-1] = 0
    D_temp = np.zeros((3,3))
    D_temp[0][0] = D[0]
    D_temp[1][1] = D[1]
    D_temp[2][2] = D[2]
    F = np.dot(np.dot(U,D_temp),Vt)
    F_final = np.dot(np.dot(np.transpose(T2), F), T1)
    return F_final
#Function to calculate epipoles and projection matrices
def calcparam(F):
    e1_t = null_space(F)#left
    e2_t = null_space(np.transpose(F))#right
    e1 = np.array([e1_t[0][0],e1_t[1][0], e1_t[2][0]])
    e2 = np.array([e2_t[0][0],e2_t[1][0], e2_t[2][0]])
    e1 = e1/e1[2]
    e2 = e2/e2[2]
    P1 = np.zeros ((3,4))
    P1[0][0] = 1
    P1[1][1] = 1
    P1[2][2] = 1
  
    ex = np.zeros((3,3))
    ex[0][1] = -1*e2[2]
    ex[0][2] = e2[1]
    ex[1][0] = e2[2]
    ex[1][2] = -1*e2[0]
    ex[2][0] = -1*e2[1]
    ex[2][1] = e2[0]
    exF = np.matmul(ex, F)

    P2 = np.zeros ((3,4))
    P2[:,0:3] = exF
    P2[:,3] = e2

    return e1, e2, P1, P2
#Function to create matrix A for world coordinate calculation
def world_mat(P1, P2, x1, x2):
    A = np.zeros((4,4))
    A[0] = np.dot (x1[0], P1[2]) - P1[0]
    A[1] = np.dot (x1[1], P1[2]) - P1[1]
    A[2] = np.dot (x2[0], P2[2]) - P2[0]
    A[3] = np.dot (x2[1], P2[2]) - P2[1]
    return A
#Error function for LM optimization
def error(param):
    P1 = np.zeros ((3,4))
    P1[0][0] = 1
    P1[1][1] = 1
    P1[2][2] = 1

    P2 = np.reshape(param,(3,4))
    error = []
    for i in range(len(m_cor1)):
        A = world_mat(P1, P2, m_cor1[i], m_cor2[i])
        u,d,vt = np.linalg.svd(A)
        v = np.transpose(vt)
        WC = v[:,-1]
        WC = WC/WC[3]
        x1_p = np.matmul(P1,WC)
        x2_p = np.matmul(P2,WC)
        x1_p = x1_p / x1_p[2]
        x2_p = x2_p / x2_p[2]      
        err_x = np.square( m_cor1[i][0] - x1_p[0])+np.square( m_cor2[i][0] - x2_p[0])
        err_y = np.square( m_cor1[i][1] - x1_p[1])+np.square( m_cor2[i][1] - x2_p[1])       
        error.append(err_x)
        error.append(err_y)

    return error
#Function to optimize Fundamental matrix
def optim_LM( P1, P2, x1, x2):
    param = np.reshape(P2, (1,12))
    sol = optimize.root(error, param , method='lm')    
    P2_ref = np.reshape(sol.x,(3,4))
    e2 = P2_ref[:,3]
    M = P2_ref[:,:3]
    ex = np.zeros((3,3))
    ex[0][1] = -1*e2[2]
    ex[0][2] = e2[1]
    ex[1][0] = e2[2]
    ex[1][2] = -1*e2[0]
    ex[2][0] = -1*e2[1]
    ex[2][1] = e2[0]
    F_refined = np.matmul(ex,M)
    return F_refined
#Function to calculate projected points using the projection matrix
def proj_points(pt, H):
    pt_new = []
    for i in range(len(pt)):
        pt_temp = [pt[i][0],pt[i][1],1]
        temp = np.dot(H, pt_temp)
        temp = temp/temp[2]
        pt_new.append(temp)
    return np.array(pt_new)
#Function to calculate homographes for the left and right input image  
def rect_homography(img1, img2, pt1, pt2, e1, e2, F, P1, P2):
    height = img1.shape[0]
    width = img1.shape[1]
    T = np.array([[1, 0, -width/2.0],[0, 1, -height/2.0],[0, 0, 1]]);
    theta = np.arctan(-1*((e2[1]-(height/2.0))/(e2[0]-(width/2.0))))
    cos = np.cos(theta)
    sin = np.sin(theta)
    f = cos*(e2[0]-(width/2.0)) - sin*(e2[1]-(height/2.0))
    G = np.identity(3)
    G[2][0] = -(1/f)
    R = np.array([[cos, -sin, 0],[sin, cos, 0],[0, 0, 1]])
    H2 = np.matmul(np.matmul(G,R),T)	
    c_pt = [width/2, height/2, 1]
    c_rect = np.matmul(H2, c_pt)
    c_rect = c_rect/c_rect[2]
    H_tran = np.identity (3)
    H_tran [0][2] = (width/2.0) - c_rect[0] 
    H_tran [1][2] = (height/2.0) - c_rect[1]     
    H2 = np.matmul(H_tran, H2)
    
    #finding homography for the left image 
    M = np.matmul(P2, np.linalg.pinv(P1))
    H1_temp = np.matmul(H2, M)
    
    pt1_hat = proj_points(pt1, H1_temp)
    pt2_hat = proj_points(pt2, H2)

    A = pt1_hat 
    b = pt2_hat[:,0]
    X =  np.matmul(np.linalg.pinv(A), np.transpose(b))

    HA = np.identity (3) 
    HA[0] = X
    H1 = np.matmul(HA, H1_temp)

    c_pt = [width/2.0, height/2.0, 1]
    c_rect = np.matmul(H1, c_pt)
    c_rect = c_rect/c_rect[2]
    H_tran = np.identity(3)
    H_tran [0][2] = (width/2.0) - c_rect[0] 
    H_tran [1][2] = (height/2.0) - c_rect[1]   
    H1 = np.matmul(H_tran, H1)
    F_t = np.matmul(np.matmul(np.linalg.inv(np.transpose(H2)),F),np.linalg.inv(H1))

    return F_t, H1, H2
#Function to create rectified images
def rectify_img(img, H, name):
    height = img.shape[0]
    width = img.shape[1]
    bound_pt = np.zeros((3,4))
    bound_pt[0][1] = width
    bound_pt[0][3] = width
    bound_pt[1][2] = height
    bound_pt[1][3] = height
    bound_pt[2] = 1
    b_proj = np.transpose(np.matmul(H,bound_pt))
    for i in range(len(b_proj)):
        b_proj[i] = b_proj[i]/b_proj[i][2]
    b_min = np.array([np.min(b_proj[:,0]), np.min(b_proj[:,1])])
    b_max = np.array([np.max(b_proj[:,0]), np.max(b_proj[:,1])])
    height_proj = b_max[1]- b_min[1]
    width_proj = b_max[0]- b_min[0]
    scale = np.array ([[width/float(width_proj), 0, 0 ],[0, height/float(height_proj) , 0 ],[0, 0, 1]])
    H_s = np.matmul(scale, H)
    b_proj = np.transpose(np.matmul(H_s,bound_pt))
    for i in range(len(b_proj)):
        b_proj[i] = b_proj[i]/b_proj[i][2]
    off_x = int(np.min(b_proj[:,0]))
    off_y = int(np.min(b_proj[:,1]))
    T = np.array([[1,0,-off_x],[0,1,-off_y],[0,0,1]])
    H_rect = np.matmul( T,H_s )
    img_new = np.zeros((img.shape[0],img.shape[1], 3), dtype ='uint8' )
    for row in range (0, img_new.shape[0]-1): 
        for col in range (0, img_new.shape[1]-1):
            temp = np.ones(3)
            temp[0] = col
            temp[1] = row
            new_rc = np.matmul(np.linalg.inv(H_rect),temp)
            new_rc = np.divide(new_rc,new_rc[2])
            if (new_rc[0]>0) and (new_rc[0]<img.shape[1])and (new_rc[1]>0) and (new_rc[1]<img.shape[0]):
                img_new[row][col] = img[int(new_rc[1]),int(new_rc[0])]  
    cv2.imwrite('%s_rectified.jpg'%(name),img_new)
#Function to find SIFT correspondence points   
def SIFT(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    i = 0
    j = 0 
    l1 = []
    l2 = []
    ll1 = []
    ll2 = []
    center = np.array([310, 250])
    far = np.array([299,55])
    point_diff = np.subtract (center, far)
    max_dist = 0.7 * np.linalg.norm (point_diff)
    print (max_dist)
    for i in range(0, len(des1)):
        dist_min = np.Inf
        for j in range (0, len(des2)):
            diff = np.subtract(des1[i], des2[j])
            sqr = np.power(diff,2)
            SSD = np.sum(sqr)
            if SSD < dist_min:
                dist_min = SSD
                pair1 = [kp1[i].pt[0],kp1[i].pt[1], SSD]
                pair2 = [kp2[j].pt[0],kp2[j].pt[1], SSD]
        new_pt1 = np.array([pair1[0],pair1[1]])
        new_pt2 = np.array([pair2[0],pair2[1]])
        pt1_diff = np.linalg.norm (np.subtract (center, new_pt1))
        pt2_diff = np.linalg.norm (np.subtract (center, new_pt2))
        if pt1_diff < max_dist and pt2_diff < max_dist:
            ll1.append(pair1)
            ll2.append(pair2)
    ll1.sort(key = takethird)
    ll2.sort(key = takethird)
    l1 = ll1 [1:40]
    l2 = ll2 [1:40]
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
        cv2.line(img_new,(int(l1[i][0]),int(l1[i][1])),(int(l2[i][0]+img1.shape[1]),int(l2[i][1])),(0,255,0),1)
    cv2.imwrite('SIFT_correspondence.jpg', img_new)
    return l1,l2
#Function to rearrage inlier into correspondence points
def rearrange(p1, p2):
    new_pt1 = []
    new_pt2 = []
    for i in range (len(p1)):
        temp = np.array ([p1[i][0], p1[i][1]])
        new_pt1.append(temp)
    for i in range (len(p2)):
        temp = np.array ([p2[i][0], p2[i][1]])
        new_pt2.append(temp)
    return new_pt1, new_pt2
#Function to calculate world coordinates
def world_cord(P1, P2, pt1, pt2):
    world_coord = []
    for i in range(len(pt1)):
        A = world_mat(P1, P2, pt1[i], pt2[i])
        u,d,vt = np.linalg.svd(A)
        v = np.transpose(vt)
        WC = v[:,-1]
        WC = WC/WC[3]
        world_coord.append([WC[0], WC[1], WC[2]])
    return world_coord
    
#Function call for creating rectified images
img1 = cv2.imread('tea_left_2.jpg') #left
img2 = cv2.imread('tea_right_2.jpg') #right

m_cor1 = np.array([[107,197],[351,295],[461,151],[247,83],[314,142],[277,157],[296,226],[176,411],[346,502],[427,368]])
m_cor2 = np.array([[112,150],[320,272],[465,145],[270,55],[320,119],[281,130],[282,199],[171,365],[317,479],[422,361]])

inlier_img(img1, img2, m_cor1, m_cor2)
norm_pt1, T1 = normalize(m_cor1)
norm_pt2, T2 = normalize(m_cor2)

F = computeF(norm_pt1, norm_pt2, T1, T2)
e1, e2, P1, P2 = calcparam(F)

F_refined = optim_LM(P1 , P2, m_cor1, m_cor2)
e1_ref, e2_ref, P1_ref, P2_ref = calcparam(F_refined)
#
F_rect, H1, H2= rect_homography(img1, img2, m_cor1, m_cor2, e1_ref, e2_ref, F_refined, P1_ref, P2_ref)
e1_rect, e2_rect, P1_rect, P2_rect = calcparam(F_rect)
##
name = 'left'
rectify_img(img1, H2,name)
name = 'right'
rectify_img(img2, H2,name)


#Function calls for 3D projection using rectified images

img3 = cv2.imread('left_rectified.jpg') #left
img4 = cv2.imread('right_rectified.jpg') #right
l1, l2 = SIFT(img3,img4)
m_cor1, m_cor2 = rearrange(l1, l2)
norm_pt1, T1 = normalize(m_cor1)
norm_pt2, T2 = normalize(m_cor2)
F = computeF(norm_pt1, norm_pt2, T1, T2)
e1, e2, P1, P2 = calcparam(F)
F_refined = optim_LM(P1 , P2, m_cor1, m_cor2)
e1_ref, e2_ref, P1_ref, P2_ref = calcparam(F_refined)

F_rect, H1, H2 = rect_homography(img3, img4, m_cor1, m_cor2, e1_ref, e2_ref, F_refined, P1_ref, P2_ref)
e1_rect, e2_rect, P1_rect, P2_rect = calcparam(F_rect)

bound1 = np.array([[160,196],[405,253],[486,128],[297,83],[249,377],[412,419],[470,499]])#
bound2 = np.array([[163,149],[373,238],[468,124],[302,55],[255,339],[397,405],[463,294]])

wc = np.divide (world_cord(P1_rect, P2_rect, m_cor1, m_cor2),1)
wc_b = np.divide (world_cord(P1_rect, P2_rect, bound1, bound2),1)

#3D plotting of world coordinates 
fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range (len(wc)):
    if i != 20 and i != 35:
        ax.scatter3D(wc[i][0], wc[i][1], wc[i][2], color='b')

for i in range (len(wc_b)):
    ax.scatter3D(wc_b[i][0],wc_b[i][1],wc_b[i][2], color='g')
   
x = [wc_b[0],wc_b[1],wc_b[2],wc_b[3],wc_b[4],wc_b[5],wc_b[6]]    
plt.plot([x[0][0],x[1][0]],[x[0][1],x[1][1]],[x[0][2],x[1][2]] , color='r')
plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],[x[1][2],x[2][2]] , color='r')
plt.plot([x[2][0],x[3][0]],[x[2][1],x[3][1]],[x[2][2],x[3][2]] , color='r')#
plt.plot([x[3][0],x[0][0]],[x[3][1],x[0][1]],[x[3][2],x[0][2]] , color='r')
plt.plot([x[0][0],x[4][0]],[x[0][1],x[4][1]],[x[0][2],x[4][2]] , color='r')
plt.plot([x[5][0],x[1][0]],[x[5][1],x[1][1]],[x[5][2],x[1][2]] , color='r')
plt.plot([x[6][0],x[5][0]],[x[6][1],x[5][1]],[x[6][2],x[5][2]] , color='r')
plt.plot([x[6][0],x[2][0]],[x[6][1],x[2][1]],[x[6][2],x[2][2]] , color='r')    
plt.plot([x[5][0],x[4][0]],[x[5][1],x[4][1]],[x[5][2],x[4][2]] , color='r')














